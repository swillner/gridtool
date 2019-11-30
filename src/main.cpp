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

#include <CImg.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include "colormaps.h"
#include "netcdftools.h"
#include "nvector.h"
#include "progressbar.h"
#include "version.h"

#undef None
#include <args.hxx>

using namespace cimg_library;

template<typename T>
static std::pair<float, float> aggregate(
    netCDF::NcVar var, std::size_t latdiv, std::size_t londiv, nvector::Vector<float, 2>& outgrid, std::size_t max_memusage, std::size_t time) {
    std::pair<float, float> minmax = {std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest()};
    bool fill_mode;
    T fill_value;
    var.getFillModeParameters(fill_mode, &fill_value);
    size_t first_dim = 0;
    if (var.getDimCount() == 3) {
        first_dim = 1;
    }
    const auto lat_count = var.getDim(first_dim).getSize();
    const auto lon_count = var.getDim(first_dim + 1).getSize();
    const auto chunk_size = std::min(max_memusage / sizeof(T) / lon_count, lat_count) / latdiv;
    nvector::Vector<T, 2> ingrid(0, chunk_size * latdiv, lon_count);
    progressbar::ProgressBar bar(lat_count / latdiv / chunk_size);
    for (std::size_t offset = 0; offset < lat_count / latdiv; offset += chunk_size) {
        const auto size = std::min(chunk_size, lat_count / latdiv - offset);
        if (first_dim == 0) {
            var.getVar({offset * latdiv, 0}, {size * latdiv, lon_count}, &ingrid.data()[0]);
        } else {
            var.getVar({time, offset * latdiv, 0}, {1, size * latdiv, lon_count}, &ingrid.data()[0]);
        }
        nvector::View<float, 2> outgrid_view(std::begin(outgrid.data()), {nvector::Slice(offset, size, lon_count / londiv), outgrid.slice<1>()});
        nvector::foreach_view_parallel(nvector::collect(outgrid_view), [&](std::size_t lat, std::size_t lon, float& d) {
            float res = 0;
            bool valid = false;
            for (auto y = lat * latdiv; y < (lat + 1) * latdiv; ++y) {
                for (auto x = lon * londiv; x < (lon + 1) * londiv; ++x) {
                    const auto v = ingrid(y, x);
                    if (v != fill_value && !std::isnan(v)) {
                        res += v;
                        valid = true;
                    }
                }
            }
            if (valid) {
#pragma omp critical
                {
                    if (res < minmax.first) {
                        minmax.first = res;
                    }
                    if (res > minmax.second) {
                        minmax.second = res;
                    }
                }
                d = res;
            }
        });
        ++bar;
    }
    bar.close(true);
    return minmax;
}

args::Group arguments("arguments");
args::HelpFlag help(arguments, "help", "Print this help text", {'h', "help"});
args::Flag version(arguments, "version", "Print version", {'v', "version"});
int main(int argc, const char** argv) {
#ifndef DEBUG
    try {
#endif
        args::ArgumentParser p(
            "gridtool"
            "\n\n"
            "Version: " GRIDTOOL_VERSION
            "\n\n"
            "Author:  Sven Willner <sven.willner@gmail.com>"
            "\n");
        args::Group commands(p, "commands");

        args::Command view(commands, "view", "view grid", [&](args::Subparser& parser) {
            args::ValueFlag<std::size_t> max_memusage_p(parser, "max_memusage", "Maximum memory usage (in MiB) [default: 1000]", {"max-memusage"}, 1000);
            args::ValueFlag<std::size_t> width_p(parser, "width", "Maximum window width [default: 1024]", {"width"}, 1024);
            args::ValueFlag<std::size_t> height_p(parser, "height", "Maximum window height [default: 768]", {"height"}, 768);
            args::ValueFlag<std::size_t> time_p(parser, "time", "Time step to view [default: 0]", {"time"}, 0);
            args::ValueFlag<std::string> varname_p(parser, "variable", "Variable to plot", {"variable"});
            args::Positional<std::string> filename_p(parser, "filename", "File to view");
            parser.Parse();

            const auto filename = filename_p.Get();
            if (filename.empty()) {
                throw args::Error("Filename required");
            }

            const auto varname = varname_p.Get();
            const auto max_memusage = max_memusage_p.Get() * 1024 * 1024;
            const auto width = width_p.Get();
            const auto height = height_p.Get();
            const auto time = time_p.Get();

            netCDF::NcFile infile(filename, netCDF::NcFile::read);
            netCDF::NcVar var;

            if (varname.empty()) {
                bool found = false;
                for (const auto v : infile.getVars()) {
                    if (v.second.getDimCount() == 2 || v.second.getDimCount() == 3) {
                        var = v.second;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    throw std::runtime_error("No viewable variable found");
                }
            } else {
                var = infile.getVar(varname);
            }

            size_t first_dim = 0;
            if (var.getDimCount() == 3) {
                first_dim = 1;
            }
            const auto lat_count = var.getDim(first_dim).getSize();
            const auto lon_count = var.getDim(first_dim + 1).getSize();
            auto latdiv = (lat_count + height - 1) / height;
            auto londiv = (lon_count + width - 1) / width;
            if (latdiv < londiv) {
                latdiv = londiv;
            } else {
                londiv = latdiv;
            }

            nvector::Vector<float, 2> outgrid(std::numeric_limits<float>::quiet_NaN(), lat_count / latdiv, lon_count / londiv);

            std::pair<float, float> minmax;
            switch (var.getType().getTypeClass()) {
                case netCDF::NcType::nc_DOUBLE:
                    minmax = aggregate<double>(var, latdiv, londiv, outgrid, max_memusage, time);
                    break;
                case netCDF::NcType::nc_FLOAT:
                    minmax = aggregate<float>(var, latdiv, londiv, outgrid, max_memusage, time);
                    break;
                case netCDF::NcType::nc_USHORT:
                    minmax = aggregate<std::int16_t>(var, latdiv, londiv, outgrid, max_memusage, time);
                    break;
                case netCDF::NcType::nc_INT:
                    minmax = aggregate<int>(var, latdiv, londiv, outgrid, max_memusage, time);
                    break;
                case netCDF::NcType::nc_BYTE:
                    minmax = aggregate<signed char>(var, latdiv, londiv, outgrid, max_memusage, time);
                    break;
                default:
                    throw std::runtime_error("Variable type not supported");
            }

            std::cout << "Min: " << minmax.first << std::endl;
            std::cout << "Max: " << minmax.second << std::endl;
            CImg<unsigned char> img(lon_count / londiv, lat_count / latdiv, 1, 3);
            nvector::foreach_view_parallel(nvector::collect(outgrid), [&](std::size_t lat, std::size_t lon, float d) {
                if (std::isnan(d)) {
                    img(lon, lat, 0, 0) = 255;
                    img(lon, lat, 0, 1) = 255;
                    img(lon, lat, 0, 2) = 255;
                } else {
                    const auto c = turbo_srgb_bytes[std::lround(255 * ((d - minmax.first) / (minmax.second - minmax.first)))];
                    img(lon, lat, 0, 0) = c[0];
                    img(lon, lat, 0, 1) = c[1];
                    img(lon, lat, 0, 2) = c[2];
                }
            });
            // img.display(0, false);
            CImgDisplay main_disp(img.width(), img.height());
            main_disp.display(img);
            while (!main_disp.is_closed()) {
                main_disp.wait();
                if (main_disp.button()) {
                    break;
                }
            }
        });
        args::GlobalOptions globals(p, arguments);

        try {
            p.ParseCLI(argc, argv);
        } catch (args::Help) {
            std::cout << p;
        } catch (args::Error& e) {
            std::cerr << e.what() << "\n\n" << p;
            return 1;
        }

        if (version) {
            std::cout << GRIDTOOL_VERSION << std::endl;
        }

        return 0;
#ifndef DEBUG
    } catch (std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return 255;
    }
#endif
}
