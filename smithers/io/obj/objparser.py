# The source code in this file is baded on
# http://jamesgregson.ca/loadsave-wavefront-obj-files-in-python.html
# I removed several features which were not needed, cleaned the code, and
# introduced the management of multiple regions of the .obj file

import numpy as np


class WavefrontOBJ:
    def __init__(self):
        self.path = None
        self.regions = []
        # contain 2-tuples such that the first item is the first polygon which
        # belongs to a new region, the second item is the index of the region
        # in self.regions
        self.regions_change_indexes = []
        self.vertices = []
        self.normals = []
        # M*Nv*3 array, Nv=# of vertices, stored as vid,tid,nid (-1 for N/A)
        self.polygons = []


def load_obj(filename: str) -> WavefrontOBJ:
    """Reads a .obj file from disk and returns a WavefrontOBJ instance. Does
    not support a lot of features at the moment.
    """

    with open(filename, "r") as objf:
        obj = WavefrontOBJ()
        obj.path = filename

        for line in objf:
            toks = line.split()

            # header
            if toks[0] == "#" and len(toks) == 3 and toks[1].isnumeric():
                obj.regions.append(toks[2])
            elif toks[0] == "v":
                obj.vertices.append([float(v) for v in toks[1:]])
            elif toks[0] == "vn":
                obj.normals.append([float(v) for v in toks[1:]])
            elif toks[0] == "f":
                obj.polygons.append(list(map(int, toks[1:])))
            elif toks[0] == "g":
                idx = len(obj.polygons)
                region_idx = obj.regions.index(toks[1])
                obj.regions_change_indexes.append((idx, region_idx))
            else:
                print("skipping: {}".format(toks))

        obj.vertices = np.array(obj.vertices)

        return obj


def generate_region_string(obj):
    return "\n".join(
        [
            "#     {}    {}".format(idx, name)
            for idx, name in enumerate(obj.regions)
        ]
    )


def generate_header(obj):
    return """# Wavefront OBJ file
# Regions:
{}
#
# points    : {}
# triangles : {}
#""".format(
        generate_region_string(obj), len(obj.vertices), len(obj.polygons)
    )


def save_obj(obj: WavefrontOBJ, path: str):
    """Saves a WavefrontOBJ object to a file

    Warning: Contains no error checking!

    """
    with open(path, "w") as ofile:
        # write header
        ofile.write(generate_header(obj) + "\n")

        for vtx in obj.vertices:
            ofile.write("v " + " ".join(map(str, vtx)) + "\n")
        for nrm in obj.normals:
            ofile.write(
                "vn " + " ".join(["{}".format(vn) for vn in nrm]) + "\n"
            )

        current_region_idx = -1

        for poly_idx, polygon in enumerate(obj.polygons):
            if (
                poly_idx
                == obj.regions_change_indexes[current_region_idx + 1][0]
            ):
                # the region changes NOW
                current_region_idx += 1
                ofile.write(
                    "g {}\n".format(
                        obj.regions[
                            obj.regions_change_indexes[current_region_idx][1]
                        ]
                    )
                )

            ofile.write("f {} {} {}".format(*polygon) + "\n")
