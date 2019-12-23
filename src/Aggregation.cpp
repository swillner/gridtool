/*
  Copyright (C) 2019 Sven Willner <sven.willner@gmail.com>

  This infile is part of gridtool.

  gridtool is free software: you can redistribute it and/or modify
  it under the terms of the GNU Affero General Public License as
  published by the Free Software Foundation, either version 3 of
  the License, or (at your option) any later version.

  gridtool is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU Affero General Public License for more details.

  You should have received a copy of the GNU Affero General Public License
  along with gridtool.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "Aggregation.h"
#include <limits>
#include "progressbar.h"

template<typename T>
MinMax<T> Aggregation<T>::aggregate(netCDF::NcVar var, bool calc_minmax) {
    MinMax<T> minmax = {std::numeric_limits<T>::max(), std::numeric_limits<T>::lowest()};
    netCDF::for_type(var.getType().getTypeClass(), [&](auto type) {
        using SourceType = decltype(type);

        // get fill value
        SourceType fill_value = std::numeric_limits<SourceType>::quiet_NaN();
        {
            bool fill_mode;
            var.getFillModeParameters(fill_mode, &fill_value);
            const auto att = var.getAtt("_FillValue");
            if (!att.isNull()) {
                att.getValues(&fill_value);
            }
        }

        const bool with_time = var.getDimCount() == 3;
        const auto lat_count = var.getDim(with_time ? 1 : 0).getSize();
        const auto lon_count = var.getDim(with_time ? 2 : 1).getSize();
        source_lat_offset = source_lat_offset % latdiv;
        source_lon_offset = source_lon_offset % londiv;
        const auto target_lat_count = (source_lat_offset + lat_count + latdiv - 1) / latdiv;
        const auto target_lon_count = (source_lon_offset + lon_count + londiv - 1) / londiv;
        outgrid.resize(std::numeric_limits<T>::quiet_NaN(), target_lat_count, target_lon_count);

        const auto chunk_lat_count = std::min(max_memusage / (sizeof(SourceType) * lon_count * latdiv), target_lat_count);
        nvector::Vector<SourceType, 2> ingrid(fill_value, chunk_lat_count * latdiv, lon_count);

        progressbar::ProgressBar bar(target_lat_count / chunk_lat_count);
        for (long chunk_lat_offset = 0; chunk_lat_offset < target_lat_count; chunk_lat_offset += chunk_lat_count) {
            if (chunk_lat_offset + chunk_lat_count >= target_lat_count) {  // last chunk
                ingrid.reset(fill_value);                                  // reset to make sure excess source grid cells are not counted from previous chunk
            }
            const auto ingrid_offset = chunk_lat_offset == 0 ? source_lat_offset * lon_count : 0;
            const auto chunk_source_lat_offset = chunk_lat_offset * latdiv - (chunk_lat_offset == 0 ? 0 : source_lat_offset);
            const auto chunk_source_lat_count =
                std::min(lat_count - chunk_source_lat_offset, chunk_lat_count * latdiv) - (chunk_lat_offset == 0 ? source_lat_offset : 0);
            if (with_time) {
                var.getVar({timestep, chunk_source_lat_offset, 0}, {1, chunk_source_lat_count, lon_count}, &ingrid.data()[ingrid_offset]);
            } else {
                var.getVar({chunk_source_lat_offset, 0}, {chunk_source_lat_count, lon_count}, &ingrid.data()[ingrid_offset]);
            }
            nvector::View<T, 2> outgrid_view(
                std::begin(outgrid.data()), {nvector::Slice(chunk_lat_offset, std::min(chunk_lat_count, target_lat_count - chunk_lat_offset), target_lon_count),
                                             outgrid.template slice<1>()});
            nvector::foreach_view_parallel(nvector::collect(outgrid_view), [&](long lat, long lon, T& d) {
                T res = 0;
                bool valid = false;
                const long lon_start = lon == 0 ? 0 : lon * londiv - source_lon_offset;
                const long lon_end = std::min((lon + 1) * londiv - source_lon_offset, lon_count);
                for (long y = lat * latdiv; y < (lat + 1) * latdiv; ++y) {
                    for (long x = lon_start; x < lon_end; ++x) {
                        const auto v = ingrid(y, x);
                        if (v > 0 && v != fill_value && !std::isnan(v)) {
                            res += v;
                            valid = true;
                        }
                    }
                }
                if (valid) {
                    d = res;
                    if (calc_minmax) {
#pragma omp critical
                        {
                            if (res < minmax.min) {
                                minmax.min = res;
                            }
                            if (res > minmax.max) {
                                minmax.max = res;
                            }
                        }
                    }
                }
            });
            ++bar;
        }
        bar.close(true);
    });
    return minmax;
}

template<typename T>
T Aggregation<T>::total(netCDF::NcVar var) {
    T sum = 0;
    netCDF::for_type(var.getType().getTypeClass(), [&](auto type) {
        using SourceType = decltype(type);

        // get fill value
        SourceType fill_value = std::numeric_limits<SourceType>::quiet_NaN();
        {
            bool fill_mode;
            var.getFillModeParameters(fill_mode, &fill_value);
            const auto att = var.getAtt("_FillValue");
            if (!att.isNull()) {
                att.getValues(&fill_value);
            }
        }

        const bool with_time = var.getDimCount() == 3;
        const auto lat_count = var.getDim(with_time ? 1 : 0).getSize();
        const auto lon_count = var.getDim(with_time ? 2 : 1).getSize();

        const auto chunk_lat_count = std::min(max_memusage / (sizeof(SourceType) * lon_count), lat_count);
        std::vector<SourceType> ingrid(chunk_lat_count * lon_count);

        progressbar::ProgressBar bar(lat_count / chunk_lat_count);
        for (std::size_t chunk_lat_offset = 0; chunk_lat_offset < lat_count; chunk_lat_offset += chunk_lat_count) {
            const auto this_chunk_lat_count = std::min(lat_count - chunk_lat_offset, chunk_lat_count);
            if (with_time) {
                var.getVar({timestep, chunk_lat_offset, 0}, {1, this_chunk_lat_count, lon_count}, &ingrid[0]);
            } else {
                var.getVar({chunk_lat_offset, 0}, {this_chunk_lat_count, lon_count}, &ingrid[0]);
            }
            for (long i = 0; i < this_chunk_lat_count * lon_count; ++i) {
                const auto v = ingrid[i];
                if (v > 0 && v != fill_value && !std::isnan(v)) {
                    sum += v;
                }
            }
            ++bar;
        }
        bar.close(true);
    });
    return sum;
}

template struct Aggregation<double>;
template struct Aggregation<float>;
template struct Aggregation<int>;
template struct Aggregation<int16_t>;
template struct Aggregation<signed char>;
