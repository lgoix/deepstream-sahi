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
 * Merges duplicate detections from sliced inference using GreedyNMM.
 */

#ifndef __GST_NVSAHIPOSTPROCESS_H__
#define __GST_NVSAHIPOSTPROCESS_H__

#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>
#include <vector>
#include "gstnvdsmeta.h"

#define PACKAGE "nvsahipostprocess"
#define VERSION "1.0"
#define LICENSE "Apache-2.0"
#define DESCRIPTION "DeepStream SAHI post-process plugin — GreedyNMM duplicate merging"
#define BINARY_PACKAGE "DeepStream SAHI Post-Process"
#define URL "https://github.com/levipereira/deepstream-sahi"

G_BEGIN_DECLS

typedef struct _GstNvSahiPostProcess GstNvSahiPostProcess;
typedef struct _GstNvSahiPostProcessClass GstNvSahiPostProcessClass;

#define GST_TYPE_NVSAHIPOSTPROCESS (gst_nvsahipostprocess_get_type())
#define GST_NVSAHIPOSTPROCESS(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_NVSAHIPOSTPROCESS, GstNvSahiPostProcess))
#define GST_NVSAHIPOSTPROCESS_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_NVSAHIPOSTPROCESS, GstNvSahiPostProcessClass))
#define GST_IS_NVSAHIPOSTPROCESS(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_NVSAHIPOSTPROCESS))
#define GST_IS_NVSAHIPOSTPROCESS_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_NVSAHIPOSTPROCESS))

typedef enum {
  SAHI_METRIC_IOU = 0,
  SAHI_METRIC_IOS = 1,
} SahiMatchMetric;

typedef struct {
  gfloat left, top, right, bottom;
  gfloat score;
  gint   class_id;
  gfloat area;
  NvDsObjectMeta *obj_meta;
  gboolean merged;
} SahiDetection;

struct _GstNvSahiPostProcess
{
  GstBaseTransform base_trans;

  gint              gie_id;
  guint             match_metric;
  gfloat            match_threshold;
  gboolean          class_agnostic;
  gboolean          enable_merge;

  std::vector<SahiDetection>  detections;
  std::vector<bool>           suppressed;
  std::vector<guint>          sorted_indices;
};

struct _GstNvSahiPostProcessClass
{
  GstBaseTransformClass parent_class;
};

GType gst_nvsahipostprocess_get_type (void);

G_END_DECLS

#endif /* __GST_NVSAHIPOSTPROCESS_H__ */
