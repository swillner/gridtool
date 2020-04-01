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
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include "Aggregation.h"
#include "GeoGrid.h"
#include "colormaps.h"
#include "netcdftools.h"
#include "nvector.h"
#include "progressbar.h"
#include "version.h"

#undef None
#include <args.hxx>

using namespace cimg_library;

// TODO Cleanup asserts

namespace netCDF {

template<typename T>
struct NetCDFType {};
template<>
struct NetCDFType<double> {
    static const netCDF::NcType::ncType type = netCDF::NcType::nc_DOUBLE;
};
template<>
struct NetCDFType<float> {
    static const netCDF::NcType::ncType type = netCDF::NcType::nc_FLOAT;
};
template<>
struct NetCDFType<std::int16_t> {
    static const netCDF::NcType::ncType type = netCDF::NcType::nc_USHORT;
};
template<>
struct NetCDFType<int> {
    static const netCDF::NcType::ncType type = netCDF::NcType::nc_INT;
};
template<>
struct NetCDFType<const char*> {
    static const netCDF::NcType::ncType type = netCDF::NcType::nc_STRING;
};
template<>
struct NetCDFType<std::string> {
    static const netCDF::NcType::ncType type = netCDF::NcType::nc_STRING;
};

}  // namespace netCDF

static netCDF::NcFile netcdf_read(const std::string& filename) {
    netCDF::check_file_exists(filename);
    return netCDF::NcFile(filename, netCDF::NcFile::read);
}

template<typename T>
class DimVar : public std::vector<T> {
  protected:
    const std::string name_m;
    std::vector<std::tuple<std::string, netCDF::NcType, std::size_t, std::vector<char>>> attributes;

  public:
    using std::vector<T>::size;

    DimVar(const netCDF::NcDim& dim, const netCDF::NcVar& var) : name_m(dim.getName()), std::vector<T>(dim.getSize()) {
        var.getVar(&(*this)[0]);
        for (const auto& att : var.getAtts()) {
            std::size_t size = att.second.getAttLength();
            std::vector<char> value(size);
            att.second.getValues(&value[0]);
            attributes.emplace_back(std::make_tuple(att.first, att.second.getType(), size, value));
        }
    }
    DimVar(const netCDF::NcFile& file, const std::string& name) : DimVar(file.getDim(name), file.getVar(name)) {}
    DimVar(const DimVar&) = delete;
    // DimVar(DimVar&&) = default;  // NOLINT(performance-noexcept-move-constructor,hicpp-noexcept-move) [cannot use noexpect here]
    netCDF::NcDim write_to(netCDF::NcFile& file) const {
        netCDF::NcDim result = file.addDim(name_m, size());
        netCDF::NcVar var = file.addVar(name_m, netCDF::NetCDFType<T>::type, {result});
        for (const auto& att : attributes) {
            var.putAtt(std::get<0>(att), std::get<1>(att), std::get<2>(att), &std::get<3>(att)[0]);
        }
        var.putVar(&(*this)[0]);
        return result;
    }
    const std::string& name() const { return name_m; }
};

static std::tuple<netCDF::NcVar, std::string, std::size_t, std::size_t> get_grid_var(netCDF::NcFile& file, std::string varname_p) {
    netCDF::NcVar var;
    std::string varname = varname_p;
    if (varname.empty()) {
        bool found = false;
        for (const auto v : file.getVars()) {
            if (v.second.getDimCount() == 2 || v.second.getDimCount() == 3) {
                varname = v.first;
                var = v.second;
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::runtime_error("No grid variable found");
        }
    } else {
        var = file.getVar(varname);
    }

    size_t first_dim = 0;
    if (var.getDimCount() == 3) {
        first_dim = 1;
    }
    const auto lat_count = var.getDim(first_dim).getSize();
    const auto lon_count = var.getDim(first_dim + 1).getSize();
    return std::make_tuple(var, varname, lat_count, lon_count);
}

args::Group arguments("arguments");
args::HelpFlag help(arguments, "help", "Print this help text", {'h', "help"});
args::Flag version(arguments, "version", "Print version", {'v', "version"});

int main(int argc, const char** argv) {
    try {
        args::ArgumentParser p(
            "gridtool"
            "\n\n"
            "Version: " GRIDTOOL_VERSION
            "\n\n"
            "Author:  Sven Willner <sven.willner@gmail.com>"
            "\n");
        args::Group commands(p, "commands");

        args::Command aggregate_cmd(commands, "aggregate", "aggregate grid", [&](args::Subparser& parser) {
            args::ValueFlag<std::size_t> max_memusage_p(parser, "max_memusage", "Maximum memory usage (in MiB) [default: 1000]", {"max-memusage"}, 1000);
            args::ValueFlag<std::size_t> time_p(parser, "time", "Time step to aggregate [default: 0]", {"time"}, 0);
            args::ValueFlag<std::string> varname_p(parser, "variable", "Variable to aggregate", {"variable"});
            args::Flag fullgrid(parser, "fullgrid", "Put aggregate on full grid", {"fullgrid"});
            args::Positional<std::string> infilename_p(parser, "input_filename", "File to aggregate");
            args::Positional<std::size_t> latdiv_p(parser, "latdiv", "Number of source_latitudes to aggregate");
            args::Positional<std::size_t> londiv_p(parser, "londiv", "Number of longitudes to aggregate");
            args::Positional<std::string> outfilename_p(parser, "output_filename", "Output file");
            parser.Parse();

            const auto infilename = infilename_p.Get();
            if (infilename.empty()) {
                throw args::Error("Input filename required");
            }

            const auto outfilename = outfilename_p.Get();
            if (outfilename.empty()) {
                throw args::Error("Output filename required");
            }

            const auto max_memusage = max_memusage_p.Get() * 1024 * 1024;
            const auto latdiv = latdiv_p.Get();
            const auto londiv = londiv_p.Get();
            assert(latdiv > 0);
            assert(londiv > 0);
            const auto time = time_p.Get();

            netCDF::check_file_exists(infilename);
            netCDF::NcFile infile(infilename, netCDF::NcFile::read);
            netCDF::NcVar var;
            std::string varname;
            std::size_t lat_count;
            std::size_t lon_count;
            std::tie(var, varname, lat_count, lon_count) = get_grid_var(infile, varname_p.Get());

            netCDF::NcFile outfile(outfilename, netCDF::NcFile::replace, netCDF::NcFile::nc4);

            std::size_t source_lat_offset = 0;
            std::size_t first_target_lat_index = 0;
            netCDF::NcDim lat_dim;
            {
                std::vector<double> source_latitudes(lat_count);
                infile.getVar("lat").getVar(&source_latitudes[0]);
                const auto source_lat_cellsize = (source_latitudes[lat_count - 1] - source_latitudes[0]) / (lat_count - 1);
                const bool north_to_south = source_lat_cellsize > 0;
                const auto source_full_grid_lat_count =
                    static_cast<std::size_t>(180. / std::abs(source_latitudes[0] - source_latitudes[lat_count - 1]) * (lat_count - 1));
                double first_target_latitude;
                std::size_t target_lat_count;
                if (fullgrid) {
                    first_target_latitude = (north_to_south ? -90 : 90) + source_lat_cellsize * latdiv / 2;
                    target_lat_count = source_full_grid_lat_count / latdiv;
                } else {
                    first_target_latitude = source_latitudes[0] + source_lat_cellsize * (latdiv - 1) / 2;
                    target_lat_count = (lat_count + latdiv - 1) / latdiv;
                }
                std::vector<double> target_latitudes(target_lat_count);
                for (int i = 0; i < target_lat_count; ++i) {
                    target_latitudes[i] = first_target_latitude + source_lat_cellsize * i * latdiv;
                }
                lat_dim = outfile.addDim("lat", target_lat_count);
                netCDF::NcVar lat_var = outfile.addVar("lat", netCDF::NcType::nc_DOUBLE, {lat_dim});
                lat_var.putAtt("standard_name", "latitude");
                lat_var.putAtt("long_name", "latitude");
                lat_var.putAtt("units", "degrees_north");
                lat_var.putAtt("axis", "Y");
                lat_var.putVar(&target_latitudes[0]);

                if (fullgrid) {
                    std::size_t first_source_lat_index;
                    if (north_to_south) {
                        first_source_lat_index = (source_full_grid_lat_count - 1) * (source_latitudes[0] + (90 - 90. / source_full_grid_lat_count))
                                                 / (180 - 180. / source_full_grid_lat_count);
                    } else {
                        first_source_lat_index = (source_full_grid_lat_count - 1) * ((90 - 90. / source_full_grid_lat_count) - source_latitudes[0])
                                                 / (180 - 180. / source_full_grid_lat_count);
                    }
                    first_target_lat_index = first_source_lat_index / latdiv;
                    source_lat_offset = first_source_lat_index % latdiv;
                }
            }

            std::size_t source_lon_offset = 0;
            std::size_t first_target_lon_index = 0;
            netCDF::NcDim lon_dim;
            {
                std::vector<double> source_longitudes(lon_count);
                infile.getVar("lon").getVar(&source_longitudes[0]);
                const auto source_lon_cellsize = (source_longitudes[lon_count - 1] - source_longitudes[0]) / (lon_count - 1);
                const auto source_full_grid_lon_count =
                    static_cast<std::size_t>(360. / std::abs(source_longitudes[0] - source_longitudes[lon_count - 1]) * (lon_count - 1));
                double first_target_lonitude;
                std::size_t target_lon_count;
                if (fullgrid) {
                    first_target_lonitude = -180 + source_lon_cellsize * londiv / 2;
                    target_lon_count = source_full_grid_lon_count / londiv;
                } else {
                    first_target_lonitude = source_longitudes[0] + source_lon_cellsize * (londiv - 1) / 2;
                    target_lon_count = (lon_count + londiv - 1) / londiv;
                }
                std::vector<double> target_longitudes(target_lon_count);
                for (int i = 0; i < target_lon_count; ++i) {
                    target_longitudes[i] = first_target_lonitude + source_lon_cellsize * i * londiv;
                }
                lon_dim = outfile.addDim("lon", target_lon_count);
                netCDF::NcVar lon_var = outfile.addVar("lon", netCDF::NcType::nc_DOUBLE, {lon_dim});
                lon_var.putAtt("standard_name", "longitude");
                lon_var.putAtt("long_name", "longitude");
                lon_var.putAtt("units", "degrees_east");
                lon_var.putAtt("axis", "X");
                lon_var.putVar(&target_longitudes[0]);

                if (fullgrid) {
                    std::size_t first_source_lon_index;
                    first_source_lon_index = (source_full_grid_lon_count - 1) * (source_longitudes[0] + (180 - 180. / source_full_grid_lon_count))
                                             / (360 - 360. / source_full_grid_lon_count);
                    first_target_lon_index = first_source_lon_index / londiv;
                    source_lon_offset = first_source_lon_index % londiv;
                }
            }

            netCDF::NcVar outvar = outfile.addVar(varname, var.getType().getTypeClass(), {lat_dim, lon_dim});
            outvar.setCompression(false, true, 7);

            netCDF::for_type(var.getType().getTypeClass(), [&](auto type) {
                using T = decltype(type);

                outvar.setFill<T>(true, std::numeric_limits<T>::quiet_NaN());
                Aggregation<T> aggregation(latdiv, londiv);
                aggregation.max_memusage = max_memusage;
                aggregation.timestep = time;
                aggregation.source_lat_offset = source_lat_offset;
                aggregation.source_lon_offset = source_lon_offset;
                aggregation.aggregate(var);
                outvar.putVar({first_target_lat_index, first_target_lon_index},
                              {aggregation.outgrid.template size<0>(), aggregation.outgrid.template size<1>()}, &aggregation.outgrid[0]);
            });
        });

        args::Command view_cmd(commands, "view", "view grid", [&](args::Subparser& parser) {
            args::ValueFlag<std::size_t> max_memusage_p(parser, "max_memusage", "Maximum memory usage (in MiB) [default: 1000]", {"max-memusage"}, 1000);
            args::ValueFlag<std::size_t> width_p(parser, "width", "Maximum window width [default: 1500]", {"width"}, 1500);
            args::ValueFlag<std::size_t> height_p(parser, "height", "Maximum window height [default: 1000]", {"height"}, 1000);
            args::ValueFlag<std::size_t> time_p(parser, "time", "Time step to view [default: 0]", {"time"}, 0);
            args::ValueFlag<std::string> varname_p(parser, "variable", "Variable to plot", {"variable"});
            args::Flag yflip(parser, "yflip", "Flip source_latitudes", {"yflip"});
            args::Positional<std::string> filename_p(parser, "filename", "File to view");
            parser.Parse();

            const auto filename = filename_p.Get();
            if (filename.empty()) {
                throw args::Error("Filename required");
            }

            const auto max_memusage = max_memusage_p.Get() * 1024 * 1024;
            const auto width = width_p.Get();
            const auto height = height_p.Get();
            const auto time = time_p.Get();

            netCDF::check_file_exists(filename);
            netCDF::NcFile infile(filename, netCDF::NcFile::read);
            netCDF::NcVar var;
            std::string varname;
            std::size_t lat_count;
            std::size_t lon_count;
            std::tie(var, varname, lat_count, lon_count) = get_grid_var(infile, varname_p.Get());

            auto latdiv = (lat_count + height - 1) / height;
            auto londiv = (lon_count + width - 1) / width;
            if (latdiv < londiv) {
                latdiv = londiv;
            } else {
                londiv = latdiv;
            }

            Aggregation<float> aggregation(latdiv, londiv);
            aggregation.max_memusage = max_memusage;
            aggregation.timestep = time;
            const auto minmax = aggregation.aggregate(var, true);

            std::cout << "Min: " << minmax.min << std::endl;
            std::cout << "Max: " << minmax.max << std::endl;
            CImg<unsigned char> img((lon_count + londiv - 1) / londiv, (lat_count + latdiv - 1) / latdiv, 1, 3);
            nvector::foreach_view_parallel(nvector::collect(aggregation.outgrid), [&](long lat, long lon, float d) {
                if (std::isnan(d)) {
                    img(lon, lat, 0, 0) = 255;
                    img(lon, lat, 0, 1) = 255;
                    img(lon, lat, 0, 2) = 255;
                } else {
                    const auto c = turbo_srgb_bytes[std::lround(255 * ((d - minmax.min) / (minmax.max - minmax.min)))];
                    img(lon, lat, 0, 0) = c[0];
                    img(lon, lat, 0, 1) = c[1];
                    img(lon, lat, 0, 2) = c[2];
                }
            });
            if (yflip) {
                img.mirror('y');
            }
            CImgDisplay main_disp(img.width(), img.height());
            main_disp.display(img);
            while (!main_disp.is_closed()) {
                main_disp.wait();
                if (main_disp.button()) {
                    break;
                }
            }
        });

        args::Command total_cmd(commands, "total", "calculate total", [&](args::Subparser& parser) {
            args::ValueFlag<std::size_t> max_memusage_p(parser, "max_memusage", "Maximum memory usage (in MiB) [default: 1000]", {"max-memusage"}, 1000);
            args::ValueFlag<std::size_t> time_p(parser, "time", "Time step to use [default: 0]", {"time"}, 0);
            args::ValueFlag<std::string> varname_p(parser, "variable", "Variable to plot", {"variable"});
            args::Positional<std::string> filename_p(parser, "filename", "Input file");
            parser.Parse();

            const auto filename = filename_p.Get();
            if (filename.empty()) {
                throw args::Error("Filename required");
            }

            const auto max_memusage = max_memusage_p.Get() * 1024 * 1024;
            const auto time = time_p.Get();

            netCDF::check_file_exists(filename);
            netCDF::NcFile infile(filename, netCDF::NcFile::read);
            netCDF::NcVar var;
            std::string varname;
            std::size_t lat_count;
            std::size_t lon_count;
            std::tie(var, varname, lat_count, lon_count) = get_grid_var(infile, varname_p.Get());

            Aggregation<double> aggregation(1, 1);
            aggregation.max_memusage = max_memusage;
            aggregation.timestep = time;
            std::cout.setf(std::ios::fixed, std::ios::floatfield);
            std::cout << "Total: " << aggregation.total(var) << std::endl;
        });

        args::Command affected_population_cmd(commands, "affected_population", "calculate affected population", [&](args::Subparser& parser) {
            args::ValueFlag<std::size_t> chunk_size_p(parser, "chunk_size", "Number of time steps to read at once [default: 10]", {"chunk-size"}, 10);
            args::Positional<std::string> population_filename_p(parser, "population_filename", "Population file");
            args::Positional<std::string> fraction_filename_p(parser, "fraction_filename", "Fraction file");
            args::Positional<std::string> iso_raster_filename_p(parser, "iso_raster_filename", "ISO raster file");
            args::Positional<std::string> outfilename_p(parser, "output_filename", "Output file");
            parser.Parse();

            const auto chunk_size = chunk_size_p.Get();
            const auto population_file = netcdf_read(population_filename_p.Get());
            const auto fraction_file = netcdf_read(fraction_filename_p.Get());
            const auto iso_raster_file = netcdf_read(iso_raster_filename_p.Get());
            netCDF::NcFile outfile(outfilename_p.Get(), netCDF::NcFile::replace, netCDF::NcFile::nc4);

            GeoGrid<double> population_grid;
            population_grid.read_from_netcdf(population_file, population_filename_p.Get());
            GeoGrid<double> fraction_grid;
            fraction_grid.read_from_netcdf(fraction_file, fraction_filename_p.Get());
            GeoGrid<double> iso_raster_grid;
            iso_raster_grid.read_from_netcdf(iso_raster_file, iso_raster_filename_p.Get());
            GeoGrid<double> common_grid;

            const auto population_var = population_file.getVar("population");
            const auto fraction_var = fraction_file.getVar("fldfrc");
            const auto iso_raster_var = iso_raster_file.getVar("iso");

            const auto population_time_count = population_var.getDim(0).getSize();
            const auto fraction_time_count = fraction_var.getDim(0).getSize();
            const auto time_count = std::min(population_time_count, fraction_time_count);
            assert(fraction_time_count <= population_time_count);
            DimVar<double> fraction_time(fraction_file, "time");

            nvector::Vector<float, 2> iso_raster_values(-1, iso_raster_grid.lat_count, iso_raster_grid.lon_count);
            iso_raster_var.getVar({0, 0}, {iso_raster_grid.lat_count, iso_raster_grid.lon_count}, &iso_raster_values.data()[0]);
            DimVar<const char*> iso_raster_index(iso_raster_file, "index");
            const auto region_count = iso_raster_index.size();

            nvector::Vector<double, 2> affected_population(0, time_count, region_count);
            nvector::Vector<double, 2> rel_affected_population(0, time_count, region_count);
            nvector::Vector<double, 2> total_population(0, time_count, region_count);

            std::size_t chunk_pos = chunk_size;
            std::vector<float> population_buffer(chunk_size * population_grid.size());
            std::vector<float> fraction_buffer(chunk_size * fraction_grid.size());

            progressbar::ProgressBar time_bar(time_count, "Time steps", true);
            for (std::size_t t = 0; t < time_count; ++t) {
                if (chunk_pos == chunk_size) {
                    population_var.getVar({t, 0, 0}, {std::min(chunk_size, time_count - t), population_grid.lat_count, population_grid.lon_count},
                                          &population_buffer[0]);
                    fraction_var.getVar({t, 0, 0}, {std::min(chunk_size, time_count - t), fraction_grid.lat_count, fraction_grid.lon_count},
                                        &fraction_buffer[0]);
                    chunk_pos = 0;
                    time_bar.reset_eta();
                }
                nvector::View<float, 2> population_values(std::begin(population_buffer) + chunk_pos * population_grid.size(), population_grid.lat_count,
                                                          population_grid.lon_count);
                nvector::View<float, 2> fraction_values(std::begin(fraction_buffer) + chunk_pos * fraction_grid.size(), fraction_grid.lat_count,
                                                        fraction_grid.lon_count);
                ++chunk_pos;

                nvector::foreach_view(common_grid_view(common_grid, GridView{iso_raster_values, iso_raster_grid}, GridView{population_values, population_grid},
                                                       GridView{fraction_values, fraction_grid}),
                                      [&](long /* lat */, long /* lon */, float i, float p, float f) {
                                          if (f > 1e10 || p <= 0 || i < 0 || std::isnan(i) || std::isnan(f) || std::isnan(p)) {
                                              return true;
                                          }
                                          affected_population(t, static_cast<int>(i)) += f * p;
                                          return true;
                                      });

                nvector::foreach_view(common_grid_view(common_grid, GridView{iso_raster_values, iso_raster_grid}, GridView{population_values, population_grid}),
                                      [&](long /* lat */, long /* lon */, float i, float p) {
                                          if (p <= 0 || i < 0 || std::isnan(i) || std::isnan(p)) {
                                              return true;
                                          }
                                          total_population(t, static_cast<int>(i)) += p;
                                          return true;
                                      });

                nvector::foreach_view(nvector::collect(affected_population.split<false, true>()(t), total_population.split<false, true>()(t),
                                                       rel_affected_population.split<false, true>()(t)),
                                      [](long /* i */, double a, double t, double& r) {
                                          r = a / t;
                                          return true;
                                      });

                ++time_bar;
            }

            auto index_dim = iso_raster_index.write_to(outfile);
            auto time_dim = fraction_time.write_to(outfile);

            netCDF::NcVar outvar;

            outvar = outfile.addVar("affected_population", netCDF::NcType::nc_DOUBLE, {time_dim, index_dim});
            outvar.setCompression(false, true, 7);
            outvar.putVar(&affected_population.data()[0]);

            outvar = outfile.addVar("total_population", netCDF::NcType::nc_DOUBLE, {time_dim, index_dim});
            outvar.setCompression(false, true, 7);
            outvar.putVar(&total_population.data()[0]);

            outvar = outfile.addVar("rel_affected_population", netCDF::NcType::nc_DOUBLE, {time_dim, index_dim});
            outvar.setCompression(false, true, 7);
            outvar.putVar(&rel_affected_population.data()[0]);
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
    } catch (std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return 255;
    }
}
