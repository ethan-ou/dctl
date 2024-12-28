/*
    This file is part of darktable,
    Copyright (C) 2020-2024 darktable developers.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    darktable is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with darktable.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "bauhaus/bauhaus.h"
#include "common/custom_primaries.h"
#include "common/math.h"
#include "common/matrices.h"
#include "develop/imageop.h"
#include "develop/imageop_gui.h"
#include "develop/openmp_maths.h"
#include "gui/gtk.h"
#include "gui/presets.h"
#include "iop/iop_api.h"

#include <gtk/gtk.h>
#include <stdlib.h>

#define MIDDLE_GREY 0.1845f

typedef enum dt_iop_sigmoid_methods_type_t
{
  DT_SIGMOID_METHOD_PER_CHANNEL = 0, // $DESCRIPTION: "per channel"
  DT_SIGMOID_METHOD_RGB_RATIO = 1,   // $DESCRIPTION: "RGB ratio"
} dt_iop_sigmoid_methods_type_t;

typedef enum dt_iop_sigmoid_base_primaries_t
{
  DT_SIGMOID_WORK_PROFILE = 0, // $DESCRIPTION: "working profile"
  DT_SIGMOID_REC2020 = 1,      // $DESCRIPTION: "Rec2020"
  DT_SIGMOID_DISPLAY_P3 = 2,   // $DESCRIPTION: "Display P3"
  DT_SIGMOID_ADOBE_RGB = 3,    // $DESCRIPTION: "Adobe RGB (compatible)"
  DT_SIGMOID_SRGB = 4,         // $DESCRIPTION: "sRGB"
} dt_iop_sigmoid_base_primaries_t;

typedef struct dt_iop_sigmoid_params_t
{
  float middle_grey_contrast;                     // $MIN: 0.1  $MAX: 10.0 $DEFAULT: 1.5 $DESCRIPTION: "contrast"
  float contrast_skewness;                        // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "skew"
  float display_white_target;                     // $MIN: 20.0  $MAX: 1600.0 $DEFAULT: 100.0 $DESCRIPTION: "target white"
  float display_black_target;                     // $MIN: 0.0  $MAX: 15.0 $DEFAULT: 0.0152 $DESCRIPTION: "target black"
  dt_iop_sigmoid_methods_type_t color_processing; // $DEFAULT: DT_SIGMOID_METHOD_PER_CHANNEL $DESCRIPTION: "color processing"
  float hue_preservation;                         // $MIN: 0.0 $MAX: 100.0 $DEFAULT: 100.0 $DESCRIPTION: "preserve hue"
  dt_iop_sigmoid_base_primaries_t base_primaries; // $DEFAULT: DT_SIGMOID_WORK_PROFILE $DESCRIPTION: "base primaries"
} dt_iop_sigmoid_params_t;

typedef struct dt_iop_sigmoid_data_t
{
  float white_target;
  float black_target;
  float paper_exposure;
  float film_fog;
  float film_power;
  float paper_power;
  dt_iop_sigmoid_methods_type_t color_processing;
  float hue_preservation;
  float inset[3];
  float rotation[3];
  float purity;
  dt_iop_sigmoid_base_primaries_t base_primaries;
} dt_iop_sigmoid_data_t;

typedef struct dt_iop_sigmoid_global_data_t
{
  int kernel_sigmoid_loglogistic_per_channel;
  int kernel_sigmoid_loglogistic_rgb_ratio;
} dt_iop_sigmoid_global_data_t;

void init_presets(dt_iop_module_so_t *self)
{

  dt_iop_sigmoid_params_t p = {0};
  p.display_white_target = 100.0f;
  p.display_black_target = 0.0152f;
  p.color_processing = DT_SIGMOID_METHOD_PER_CHANNEL;

  const float DEG_TO_RAD = M_PI_F / 180.f;

  // smooth - a preset that utilizes the primaries feature
  p.middle_grey_contrast = 1.5f;
  // Allow a little bit more room for the highlights
  p.contrast_skewness = -0.2f;
  p.color_processing = DT_SIGMOID_METHOD_PER_CHANNEL;
  // Allow shifts of the chromaticity. This will work well for sunsets etc.
  p.hue_preservation = 0.0f;
  p.red_inset = 0.1f;
  p.green_inset = 0.1f;
  p.blue_inset = 0.15f;
  p.red_rotation = 2.f * DEG_TO_RAD;
  p.green_rotation = -1.f * DEG_TO_RAD;
  p.blue_rotation = -3.f * DEG_TO_RAD;
  // Don't restore purity - try to avoid posterization.
  p.purity = 0.f;
  // Constant base primaries (not dependent on work profile) to
  // maintain a consistent behavior
  p.base_primaries = DT_SIGMOID_REC2020;
  dt_gui_presets_add_generic(_("smooth"), self->op, self->version(),
                             &p, sizeof(p), 1, DEVELOP_BLEND_CS_RGB_SCENE);
}

// Declared here as it is used in the commit params function
DT_OMP_DECLARE_SIMD(uniform(magnitude, paper_exp, film_fog, film_power, paper_power))
static inline float _generalized_loglogistic_sigmoid(const float value,
                                                     const float magnitude,
                                                     const float paper_exp,
                                                     const float film_fog,
                                                     const float film_power,
                                                     const float paper_power)
{
  const float clamped_value = fmaxf(value, 0.0f);
  // The following equation can be derived as a model for film + paper but it has a pole at 0
  // magnitude * powf(1.0f + paper_exp * powf(film_fog + value, -film_power), -paper_power);
  // Rewritten on a stable around zero form:
  const float film_response = powf(film_fog + clamped_value, film_power);
  const float paper_response = magnitude * powf(film_response / (paper_exp + film_response), paper_power);

  // Safety check for very large floats that cause numerical errors
  return dt_isnan(paper_response) ? magnitude : paper_response;
}

void commit_params(dt_iop_module_t *self,
                   dt_iop_params_t *p1,
                   dt_dev_pixelpipe_t *pipe,
                   dt_dev_pixelpipe_iop_t *piece)
{
  dt_iop_sigmoid_params_t *params = (dt_iop_sigmoid_params_t *)p1;
  dt_iop_sigmoid_data_t *module_data = (dt_iop_sigmoid_data_t *)piece->data;
  /* Calculate actual skew log logistic parameters to fulfill the following:
   * f(scene_zero) = display_black_target
   * f(scene_grey) = MIDDLE_GREY
   * f(scene_inf)  = display_white_target
   * Slope at scene_grey independent of skewness i.e. only changed by the contrast parameter.
   */

  // Calculate a reference slope for no skew and a normalized display
  const float ref_film_power = params->middle_grey_contrast;
  const float ref_paper_power = 1.0f;
  const float ref_magnitude = 1.0f;
  const float ref_film_fog = 0.0f;
  const float ref_paper_exposure = powf(ref_film_fog + MIDDLE_GREY, ref_film_power) * ((ref_magnitude / MIDDLE_GREY) - 1.0f);
  const float delta = 1e-6f;
  const float ref_slope = (_generalized_loglogistic_sigmoid(MIDDLE_GREY + delta, ref_magnitude, ref_paper_exposure, ref_film_fog,
                                                            ref_film_power, ref_paper_power) -
                           _generalized_loglogistic_sigmoid(MIDDLE_GREY - delta, ref_magnitude, ref_paper_exposure, ref_film_fog,
                                                            ref_film_power, ref_paper_power)) /
                          2.0f / delta;

  // Add skew
  module_data->paper_power = powf(5.0f, -params->contrast_skewness);

  // Slope at low film power
  const float temp_film_power = 1.0f;
  const float temp_white_target = 0.01f * params->display_white_target;
  const float temp_white_grey_relation = powf(temp_white_target / MIDDLE_GREY, 1.0f / module_data->paper_power) - 1.0f;
  const float temp_paper_exposure = powf(MIDDLE_GREY, temp_film_power) * temp_white_grey_relation;
  const float temp_slope = (_generalized_loglogistic_sigmoid(MIDDLE_GREY + delta, temp_white_target, temp_paper_exposure,
                                                             ref_film_fog, temp_film_power, module_data->paper_power) -
                            _generalized_loglogistic_sigmoid(MIDDLE_GREY - delta, temp_white_target, temp_paper_exposure,
                                                             ref_film_fog, temp_film_power, module_data->paper_power)) /
                           2.0f / delta;

  // Figure out what film power fulfills the target slope
  // (linear when assuming display_black = 0.0)
  module_data->film_power = ref_slope / temp_slope;

  // Calculate the other parameters now that both film and paper power is known
  module_data->white_target = 0.01f * params->display_white_target;
  module_data->black_target = 0.01f * params->display_black_target;
  const float white_grey_relation = powf(module_data->white_target / MIDDLE_GREY, 1.0f / module_data->paper_power) - 1.0f;
  const float white_black_relation = powf(module_data->black_target / module_data->white_target, -1.0f / module_data->paper_power) - 1.0f;

  module_data->film_fog = MIDDLE_GREY * powf(white_grey_relation, 1.0f / module_data->film_power) / (powf(white_black_relation, 1.0f / module_data->film_power) - powf(white_grey_relation, 1.0f / module_data->film_power));
  module_data->paper_exposure = powf(module_data->film_fog + MIDDLE_GREY, module_data->film_power) * white_grey_relation;

  module_data->color_processing = params->color_processing;
  module_data->hue_preservation = fminf(fmaxf(0.01f * params->hue_preservation, 0.0f), 1.0f);

  module_data->base_primaries = params->base_primaries;
}

static void _calculate_adjusted_primaries(const dt_iop_sigmoid_data_t *const module_data,
                                          const dt_iop_order_iccprofile_info_t *const pipe_work_profile,
                                          const dt_iop_order_iccprofile_info_t *const base_profile,
                                          dt_colormatrix_t pipe_to_base,
                                          dt_colormatrix_t base_to_rendering,
                                          dt_colormatrix_t rendering_to_pipe)
{
  // Make adjusted primaries for generating the inset matrix
  //
  // References:
  // AgX by Troy Sobotka - https://github.com/sobotka/AgX-S2O3
  // Related discussions on Blender Artists forums -
  // https://blenderartists.org/t/feedback-development-filmic-baby-step-to-a-v2/1361663
  //
  // The idea is to "inset" the work RGB data toward achromatic
  // along spectral lines before per-channel curves. This makes
  // handling of bright, saturated colors much better as the
  // per-channel process desaturates them.
  // The primaries are also rotated to compensate for Abney etc.
  // and achieve a favourable shift towards yellow.

  // First, calculate matrix to get from pipe work profile to "base primaries".
  dt_colormatrix_t base_to_pipe;
  if (pipe_work_profile != base_profile)
  {
    dt_colormatrix_mul(pipe_to_base, pipe_work_profile->matrix_in_transposed,
                       base_profile->matrix_out_transposed);
    mat3SSEinv(base_to_pipe, pipe_to_base);
  }
  else
  {
    // Special case: if pipe and base profile are the same, pipe_to_base is an identity matrix.
    for (size_t i = 0; i < 4; i++)
    {
      for (size_t j = 0; j < 4; j++)
      {
        if (i == j && i < 3)
        {
          pipe_to_base[i][j] = 1.f;
          base_to_pipe[i][j] = 1.f;
        }
        else
        {
          pipe_to_base[i][j] = 0.f;
          base_to_pipe[i][j] = 0.f;
        }
      }
    }
  }

  // Rotated, scaled primaries are calculated based on the "base profile"
  float custom_primaries[3][2];
  for (size_t i = 0; i < 3; i++)
    dt_rotate_and_scale_primary(base_profile, 1.f - module_data->inset[i], module_data->rotation[i], i,
                                custom_primaries[i]);

  dt_colormatrix_t custom_to_XYZ;
  dt_make_transposed_matrices_from_primaries_and_whitepoint(custom_primaries, base_profile->whitepoint,
                                                            custom_to_XYZ);
  dt_colormatrix_mul(base_to_rendering, custom_to_XYZ, base_profile->matrix_out_transposed);

  for (size_t i = 0; i < 3; i++)
  {
    const float scaling = 1.f - module_data->purity * module_data->inset[i];
    dt_rotate_and_scale_primary(base_profile, scaling, module_data->rotation[i], i, custom_primaries[i]);
  }

  dt_make_transposed_matrices_from_primaries_and_whitepoint(custom_primaries, base_profile->whitepoint,
                                                            custom_to_XYZ);
  dt_colormatrix_t tmp;
  dt_colormatrix_mul(tmp, custom_to_XYZ, base_profile->matrix_out_transposed);
  dt_colormatrix_t rendering_to_base;
  mat3SSEinv(rendering_to_base, tmp);
  dt_colormatrix_mul(rendering_to_pipe, rendering_to_base, base_to_pipe);
}

static dt_colorspaces_color_profile_type_t _get_base_profile_type(const dt_iop_sigmoid_base_primaries_t base_primaries)
{
  if (base_primaries == DT_SIGMOID_SRGB)
    return DT_COLORSPACE_SRGB;

  if (base_primaries == DT_SIGMOID_DISPLAY_P3)
    return DT_COLORSPACE_DISPLAY_P3;

  if (base_primaries == DT_SIGMOID_ADOBE_RGB)
    return DT_COLORSPACE_ADOBERGB;

  return DT_COLORSPACE_LIN_REC2020;
}

static const dt_iop_order_iccprofile_info_t *_get_base_profile(struct dt_develop_t *dev,
                                                               const dt_iop_order_iccprofile_info_t *pipe_work_profile,
                                                               const dt_iop_sigmoid_base_primaries_t base_primaries)
{
  if (base_primaries == DT_SIGMOID_WORK_PROFILE)
    return pipe_work_profile;

  return dt_ioppr_add_profile_info_to_list(dev, _get_base_profile_type(base_primaries), "", DT_INTENT_RELATIVE_COLORIMETRIC);
}

DT_OMP_DECLARE_SIMD()
static inline void _desaturate_negative_values(const dt_aligned_pixel_t pix_in, dt_aligned_pixel_t pix_out)
{
  const float pixel_average = fmaxf((pix_in[0] + pix_in[1] + pix_in[2]) / 3.0f, 0.0f);
  const float min_value = fminf(fminf(pix_in[0], pix_in[1]), pix_in[2]);
  const float saturation_factor = min_value < 0.0f ? -pixel_average / (min_value - pixel_average) : 1.0f;
  for_each_channel(c, aligned(pix_in, pix_out))
  {
    pix_out[c] = pixel_average + saturation_factor * (pix_in[c] - pixel_average);
  }
}

typedef struct dt_iop_sigmoid_value_order_t
{
  size_t min;
  size_t mid;
  size_t max;
} dt_iop_sigmoid_value_order_t;

static void _pixel_channel_order(const dt_aligned_pixel_t pix_in, dt_iop_sigmoid_value_order_t *pixel_value_order)
{
  if (pix_in[0] >= pix_in[1])
  {
    if (pix_in[1] > pix_in[2])
    { // Case 1: r >= g >  b
      pixel_value_order->max = 0;
      pixel_value_order->mid = 1;
      pixel_value_order->min = 2;
    }
    else if (pix_in[2] > pix_in[0])
    { // Case 2: b >  r >= g
      pixel_value_order->max = 2;
      pixel_value_order->mid = 0;
      pixel_value_order->min = 1;
    }
    else if (pix_in[2] > pix_in[1])
    { // Case 3: r >= b >  g
      pixel_value_order->max = 0;
      pixel_value_order->mid = 2;
      pixel_value_order->min = 1;
    }
    else
    { // Case 4: r == g == b
      // No change of the middle value, just assign something.
      pixel_value_order->max = 0;
      pixel_value_order->mid = 1;
      pixel_value_order->min = 2;
    }
  }
  else
  {
    if (pix_in[0] >= pix_in[2])
    { // Case 5: g >  r >= b
      pixel_value_order->max = 1;
      pixel_value_order->mid = 0;
      pixel_value_order->min = 2;
    }
    else if (pix_in[2] > pix_in[1])
    { // Case 6: b >  g >  r
      pixel_value_order->max = 2;
      pixel_value_order->mid = 1;
      pixel_value_order->min = 0;
    }
    else
    { // Case 7: g >= b >  r
      pixel_value_order->max = 1;
      pixel_value_order->mid = 2;
      pixel_value_order->min = 0;
    }
  }
}

// Linear interpolation of hue that also preserve sum of channels
// Assumes hue_preservation strictly in range [0, 1]
static inline void _preserve_hue_and_energy(const dt_aligned_pixel_t pix_in,
                                            const dt_aligned_pixel_t per_channel,
                                            dt_aligned_pixel_t pix_out,
                                            const dt_iop_sigmoid_value_order_t order,
                                            const float hue_preservation)
{
  // Naive Hue correction of the middle channel
  const float chroma = pix_in[order.max] - pix_in[order.min];
  const float midscale = chroma != 0.f ? (pix_in[order.mid] - pix_in[order.min]) / chroma : 0.f;
  const float full_hue_correction = per_channel[order.min] + (per_channel[order.max] - per_channel[order.min]) * midscale;
  const float naive_hue_mid = (1.0f - hue_preservation) * per_channel[order.mid] + hue_preservation * full_hue_correction;

  const float per_channel_energy = per_channel[0] + per_channel[1] + per_channel[2];
  const float naive_hue_energy = per_channel[order.min] + naive_hue_mid + per_channel[order.max];
  const float pix_in_min_plus_mid = pix_in[order.min] + pix_in[order.mid];
  const float blend_factor = pix_in_min_plus_mid != 0.f ? 2.0f * pix_in[order.min] / pix_in_min_plus_mid : 0.f;
  const float energy_target = blend_factor * per_channel_energy + (1.0f - blend_factor) * naive_hue_energy;

  // Preserve hue constrained to maintain the same energy as the per channel result
  if (naive_hue_mid <= per_channel[order.mid])
  {
    const float corrected_mid = ((1.0f - hue_preservation) * per_channel[order.mid] + hue_preservation * (midscale * per_channel[order.max] + (1.0f - midscale) * (energy_target - per_channel[order.max]))) / (1.0f + hue_preservation * (1.0f - midscale));
    pix_out[order.min] = energy_target - per_channel[order.max] - corrected_mid;
    pix_out[order.mid] = corrected_mid;
    pix_out[order.max] = per_channel[order.max];
  }
  else
  {
    const float corrected_mid = ((1.0f - hue_preservation) * per_channel[order.mid] + hue_preservation * (per_channel[order.min] * (1.0f - midscale) + midscale * (energy_target - per_channel[order.min]))) / (1.0f + hue_preservation * midscale);
    pix_out[order.min] = per_channel[order.min];
    pix_out[order.mid] = corrected_mid;
    pix_out[order.max] = energy_target - per_channel[order.min] - corrected_mid;
  }
}

void process_loglogistic_per_channel(struct dt_develop_t *dev,
                                     dt_dev_pixelpipe_iop_t *piece,
                                     const void *const ivoid, void *const ovoid,
                                     const dt_iop_roi_t *const roi_in,
                                     const dt_iop_roi_t *const roi_out)
{
  const dt_iop_sigmoid_data_t *module_data = (dt_iop_sigmoid_data_t *)piece->data;

  const float *const in = (const float *)ivoid;
  float *const out = (float *)ovoid;
  const size_t npixels = (size_t)roi_in->width * roi_in->height;

  const float white_target = module_data->white_target;
  const float paper_exp = module_data->paper_exposure;
  const float film_fog = module_data->film_fog;
  const float contrast_power = module_data->film_power;
  const float skew_power = module_data->paper_power;
  const float hue_preservation = module_data->hue_preservation;

  const dt_iop_order_iccprofile_info_t *pipe_work_profile = dt_ioppr_get_pipe_work_profile_info(piece->pipe);
  const dt_iop_order_iccprofile_info_t *base_profile = _get_base_profile(dev, pipe_work_profile, module_data->base_primaries);
  dt_colormatrix_t pipe_to_base, base_to_rendering, rendering_to_pipe;
  _calculate_adjusted_primaries(module_data, pipe_work_profile, base_profile, pipe_to_base, base_to_rendering, rendering_to_pipe);

  DT_OMP_FOR()
  for (size_t k = 0; k < 4 * npixels; k += 4)
  {
    const float *const restrict pix_in = in + k;
    float *const restrict pix_out = out + k;
    dt_aligned_pixel_t pix_in_base, pix_in_strict_positive;
    dt_aligned_pixel_t per_channel;

    // Convert to "base primaries"
    dt_apply_transposed_color_matrix(pix_in, pipe_to_base, pix_in_base);

    // Force negative values to zero
    _desaturate_negative_values(pix_in_base, pix_in_strict_positive);

    dt_aligned_pixel_t rendering_RGB;
    dt_apply_transposed_color_matrix(pix_in_strict_positive, base_to_rendering, rendering_RGB);

    for_each_channel(c, aligned(rendering_RGB, per_channel))
    {
      per_channel[c] = _generalized_loglogistic_sigmoid(rendering_RGB[c], white_target, paper_exp, film_fog,
                                                        contrast_power, skew_power);
    }

    // Hue correction by scaling the middle value relative to the max and min values.
    dt_iop_sigmoid_value_order_t pixel_value_order;
    dt_aligned_pixel_t per_channel_hue_corrected;
    _pixel_channel_order(rendering_RGB, &pixel_value_order);
    _preserve_hue_and_energy(rendering_RGB, per_channel, per_channel_hue_corrected, pixel_value_order,
                             hue_preservation);
    dt_apply_transposed_color_matrix(per_channel_hue_corrected, rendering_to_pipe, pix_out);

    // Copy over the alpha channel
    pix_out[3] = pix_in[3];
  }
}