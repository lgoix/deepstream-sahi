/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Levi Pereira <levi.pereira@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * DeepStream SAHI Post-Process Plugin
 *
 * Sits between nvinfer and nvtracker. Merges duplicate detections produced by
 * sliced inference using the GreedyNMM algorithm (ported from SAHI's
 * combine.py). Operates entirely on NvDsObjectMeta — no tensor access, no CUDA.
 *
 * Pipeline position:
 *   nvinfer → queue → nvsahipostprocess → nvtracker → nvdsosd
 */

#include <string.h>
#include <algorithm>
#include <numeric>
#include "gstnvsahipostprocess.h"

GST_DEBUG_CATEGORY_STATIC (gst_nvsahipostprocess_debug);
#define GST_CAT_DEFAULT gst_nvsahipostprocess_debug

enum
{
  PROP_0,
  PROP_GIE_ID,
  PROP_MATCH_METRIC,
  PROP_MATCH_THRESHOLD,
  PROP_CLASS_AGNOSTIC,
  PROP_ENABLE_MERGE,
};

#define DEFAULT_GIE_ID          -1
#define DEFAULT_MATCH_METRIC    SAHI_METRIC_IOS
#define DEFAULT_MATCH_THRESHOLD 0.5f
#define DEFAULT_CLASS_AGNOSTIC  FALSE
#define DEFAULT_ENABLE_MERGE    TRUE

#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"
static GstStaticPadTemplate gst_nvsahipostprocess_sink_template =
GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM, "{ NV12, RGBA, I420 }")));

static GstStaticPadTemplate gst_nvsahipostprocess_src_template =
GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM, "{ NV12, RGBA, I420 }")));

#define gst_nvsahipostprocess_parent_class parent_class
G_DEFINE_TYPE (GstNvSahiPostProcess, gst_nvsahipostprocess,
    GST_TYPE_BASE_TRANSFORM);

static void gst_nvsahipostprocess_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * pspec);
static void gst_nvsahipostprocess_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec);
static GstFlowReturn gst_nvsahipostprocess_transform_ip (
    GstBaseTransform * btrans, GstBuffer * inbuf);

/* ── Overlap computation ─────────────────────────────────────────────────── */

static inline gfloat
compute_overlap (guint metric,
    const SahiDetection &a, const SahiDetection &b)
{
  gfloat inter_left   = MAX (a.left,   b.left);
  gfloat inter_top    = MAX (a.top,    b.top);
  gfloat inter_right  = MIN (a.right,  b.right);
  gfloat inter_bottom = MIN (a.bottom, b.bottom);

  if (inter_right <= inter_left || inter_bottom <= inter_top)
    return 0.0f;

  gfloat inter_area = (inter_right - inter_left) * (inter_bottom - inter_top);

  if (metric == SAHI_METRIC_IOU) {
    gfloat union_area = a.area + b.area - inter_area;
    return (union_area > 0.0f) ? inter_area / union_area : 0.0f;
  } else {
    gfloat min_area = MIN (a.area, b.area);
    return (min_area > 0.0f) ? inter_area / min_area : 0.0f;
  }
}

/* ── GreedyNMM core ─────────────────────────────────────────────────────── */

static void
greedy_nmm (GstNvSahiPostProcess *self,
    std::vector<SahiDetection> &dets,
    const std::vector<guint> &order,
    std::vector<bool> &suppressed)
{
  const gfloat threshold = self->match_threshold;
  const gboolean agnostic = self->class_agnostic;
  const gboolean merge = self->enable_merge;

  for (guint ii = 0; ii < order.size (); ii++) {
    guint i = order[ii];
    if (suppressed[i])
      continue;

    for (guint jj = ii + 1; jj < order.size (); jj++) {
      guint j = order[jj];
      if (suppressed[j])
        continue;

      if (!agnostic && dets[i].class_id != dets[j].class_id)
        continue;

      gfloat overlap = compute_overlap (self->match_metric, dets[i], dets[j]);

      if (overlap >= threshold) {
        suppressed[j] = true;

        if (merge) {
          dets[i].left   = MIN (dets[i].left,   dets[j].left);
          dets[i].top    = MIN (dets[i].top,    dets[j].top);
          dets[i].right  = MAX (dets[i].right,  dets[j].right);
          dets[i].bottom = MAX (dets[i].bottom, dets[j].bottom);
          dets[i].score  = MAX (dets[i].score,  dets[j].score);
          dets[i].area   = (dets[i].right - dets[i].left) *
                           (dets[i].bottom - dets[i].top);
          dets[i].merged = TRUE;
        }
      }
    }
  }
}

/* ── Per-frame processing ────────────────────────────────────────────────── */

static void
process_frame (GstNvSahiPostProcess *self,
    NvDsBatchMeta *batch_meta, NvDsFrameMeta *frame_meta)
{
  auto &dets       = self->detections;
  auto &suppressed = self->suppressed;
  auto &order      = self->sorted_indices;

  dets.clear ();

  for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL;
      l_obj = l_obj->next) {
    NvDsObjectMeta *obj = (NvDsObjectMeta *) l_obj->data;

    if (self->gie_id >= 0 &&
        obj->unique_component_id != self->gie_id)
      continue;

    SahiDetection det;
    det.left     = obj->rect_params.left;
    det.top      = obj->rect_params.top;
    det.right    = det.left + obj->rect_params.width;
    det.bottom   = det.top + obj->rect_params.height;
    det.score    = obj->confidence;
    det.class_id = obj->class_id;
    det.obj_meta = obj;
    det.area     = obj->rect_params.width * obj->rect_params.height;
    det.merged   = FALSE;
    dets.push_back (det);
  }

  if (dets.size () <= 1)
    return;

  order.resize (dets.size ());
  std::iota (order.begin (), order.end (), 0);
  std::sort (order.begin (), order.end (),
      [&dets](guint a, guint b) {
        return dets[a].score > dets[b].score;
      });

  suppressed.assign (dets.size (), false);

  greedy_nmm (self, dets, order, suppressed);

  nvds_acquire_meta_lock (batch_meta);

  for (guint i = 0; i < dets.size (); i++) {
    if (suppressed[i]) {
      nvds_remove_obj_meta_from_frame (frame_meta, dets[i].obj_meta);
    } else if (dets[i].merged) {
      NvDsObjectMeta *obj = dets[i].obj_meta;
      obj->rect_params.left   = dets[i].left;
      obj->rect_params.top    = dets[i].top;
      obj->rect_params.width  = dets[i].right - dets[i].left;
      obj->rect_params.height = dets[i].bottom - dets[i].top;

      obj->detector_bbox_info.org_bbox_coords.left   = dets[i].left;
      obj->detector_bbox_info.org_bbox_coords.top    = dets[i].top;
      obj->detector_bbox_info.org_bbox_coords.width  = obj->rect_params.width;
      obj->detector_bbox_info.org_bbox_coords.height = obj->rect_params.height;

      if (dets[i].score > obj->confidence)
        obj->confidence = dets[i].score;
    }
  }

  nvds_release_meta_lock (batch_meta);
}

/* ── GstBaseTransform vmethod ────────────────────────────────────────────── */

static GstFlowReturn
gst_nvsahipostprocess_transform_ip (GstBaseTransform * btrans,
    GstBuffer * inbuf)
{
  GstNvSahiPostProcess *self = GST_NVSAHIPOSTPROCESS (btrans);

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (inbuf);
  if (!batch_meta)
    return GST_FLOW_OK;

  for (NvDsMetaList *l_frame = batch_meta->frame_meta_list;
      l_frame != NULL; l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;
    process_frame (self, batch_meta, frame_meta);
  }

  return GST_FLOW_OK;
}

/* ── Class / instance init ───────────────────────────────────────────────── */

static void
gst_nvsahipostprocess_class_init (GstNvSahiPostProcessClass * klass)
{
  GObjectClass *gobject_class = (GObjectClass *) klass;
  GstElementClass *gstelement_class = (GstElementClass *) klass;
  GstBaseTransformClass *btrans_class = (GstBaseTransformClass *) klass;

  gobject_class->set_property =
      GST_DEBUG_FUNCPTR (gst_nvsahipostprocess_set_property);
  gobject_class->get_property =
      GST_DEBUG_FUNCPTR (gst_nvsahipostprocess_get_property);

  btrans_class->transform_ip =
      GST_DEBUG_FUNCPTR (gst_nvsahipostprocess_transform_ip);

  g_object_class_install_property (gobject_class, PROP_GIE_ID,
      g_param_spec_int ("gie-id", "GIE Unique ID",
          "Only merge detections from this GIE unique-id. "
          "-1 = merge all detections regardless of source GIE.",
          -1, G_MAXINT, DEFAULT_GIE_ID,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_MATCH_METRIC,
      g_param_spec_uint ("match-metric", "Match Metric",
          "Overlap metric: 0 = IoU (Intersection over Union), "
          "1 = IoS (Intersection over Smaller). "
          "IoS is recommended for SAHI slice-boundary merging.",
          0, 1, DEFAULT_MATCH_METRIC,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_MATCH_THRESHOLD,
      g_param_spec_float ("match-threshold", "Match Threshold",
          "Overlap threshold above which detections are considered duplicates.",
          0.0f, 1.0f, DEFAULT_MATCH_THRESHOLD,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_CLASS_AGNOSTIC,
      g_param_spec_boolean ("class-agnostic", "Class Agnostic",
          "If TRUE, match detections across different class IDs. "
          "If FALSE, only match within the same class.",
          DEFAULT_CLASS_AGNOSTIC,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_ENABLE_MERGE,
      g_param_spec_boolean ("enable-merge", "Enable Merge",
          "If TRUE, merge suppressed boxes into survivors (GreedyNMM). "
          "If FALSE, purely suppress duplicates (standard NMS).",
          DEFAULT_ENABLE_MERGE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_nvsahipostprocess_src_template));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_nvsahipostprocess_sink_template));

  gst_element_class_set_details_simple (gstelement_class,
      "SAHI Post-Process (GreedyNMM)",
      "Filter/Metadata",
      "Merges duplicate detections from SAHI sliced inference "
      "using GreedyNMM with IoU/IoS metrics",
      "Levi Pereira <levi.pereira@gmail.com>");
}

static void
gst_nvsahipostprocess_init (GstNvSahiPostProcess * self)
{
  GstBaseTransform *btrans = GST_BASE_TRANSFORM (self);

  gst_base_transform_set_in_place (btrans, TRUE);
  gst_base_transform_set_passthrough (btrans, TRUE);

  self->gie_id          = DEFAULT_GIE_ID;
  self->match_metric    = DEFAULT_MATCH_METRIC;
  self->match_threshold = DEFAULT_MATCH_THRESHOLD;
  self->class_agnostic  = DEFAULT_CLASS_AGNOSTIC;
  self->enable_merge    = DEFAULT_ENABLE_MERGE;
}

/* ── Property accessors ──────────────────────────────────────────────────── */

static void
gst_nvsahipostprocess_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstNvSahiPostProcess *self = GST_NVSAHIPOSTPROCESS (object);
  switch (prop_id) {
    case PROP_GIE_ID:
      self->gie_id = g_value_get_int (value);
      break;
    case PROP_MATCH_METRIC:
      self->match_metric = g_value_get_uint (value);
      break;
    case PROP_MATCH_THRESHOLD:
      self->match_threshold = g_value_get_float (value);
      break;
    case PROP_CLASS_AGNOSTIC:
      self->class_agnostic = g_value_get_boolean (value);
      break;
    case PROP_ENABLE_MERGE:
      self->enable_merge = g_value_get_boolean (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_nvsahipostprocess_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstNvSahiPostProcess *self = GST_NVSAHIPOSTPROCESS (object);
  switch (prop_id) {
    case PROP_GIE_ID:
      g_value_set_int (value, self->gie_id);
      break;
    case PROP_MATCH_METRIC:
      g_value_set_uint (value, self->match_metric);
      break;
    case PROP_MATCH_THRESHOLD:
      g_value_set_float (value, self->match_threshold);
      break;
    case PROP_CLASS_AGNOSTIC:
      g_value_set_boolean (value, self->class_agnostic);
      break;
    case PROP_ENABLE_MERGE:
      g_value_set_boolean (value, self->enable_merge);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/* ── Plugin registration ─────────────────────────────────────────────────── */

static gboolean
nvsahipostprocess_plugin_init (GstPlugin * plugin)
{
  GST_DEBUG_CATEGORY_INIT (gst_nvsahipostprocess_debug,
      "nvsahipostprocess", 0, "SAHI post-process (GreedyNMM) plugin");

  return gst_element_register (plugin, "nvsahipostprocess",
      GST_RANK_PRIMARY, GST_TYPE_NVSAHIPOSTPROCESS);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_sahipostprocess,
    DESCRIPTION,
    nvsahipostprocess_plugin_init,
    "1.0",
    LICENSE,
    BINARY_PACKAGE,
    URL)
