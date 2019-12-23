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

#ifndef AGGREGATION_H
#define AGGREGATION_H

#include "netcdftools.h"
#include "nvector.h"

template<typename T>
struct MinMax {
    T min;
    T max;
};

template<typename T>
struct Aggregation {
    nvector::Vector<T, 2> outgrid = {};
    std::size_t latdiv;
    std::size_t source_lat_offset = 0;
    std::size_t londiv;
    std::size_t source_lon_offset = 0;
    std::size_t max_memusage = 1000 * 1024 * 1024;
    std::size_t timestep = 0;

    Aggregation(std::size_t latdiv_p, std::size_t londiv_p) : latdiv(latdiv_p), londiv(londiv_p) {}
    MinMax<T> aggregate(netCDF::NcVar var, bool calc_minmax = false);
    T total(netCDF::NcVar var);
};

#endif
