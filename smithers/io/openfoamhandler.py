import Ofpp
import os
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter


class OpenFoamHandler:
    """
    Handler for OpenFOAM output files, based on the project Ofpp.
    """

    @classmethod
    def _build_boundary(cls, mesh, key):
        data = mesh.boundary[key]

        # extract faces which compose the boundary
        bd_faces_indexes = np.array(list(range(data.start, data.start + data.num)))
        bd_faces = np.concatenate([mesh.faces[idx] for idx in bd_faces_indexes])

        # extract a list of unique points which compose the boundary
        bd_points = np.unique(bd_faces)

        return {"faces": bd_faces_indexes, "points": bd_points, "type": data.type}

    @classmethod
    def _build_cells(cls, mesh, cell_idx):
        cell_faces_idxes = mesh.cell_faces[cell_idx]

        cell_faces = np.array([mesh.faces[idx] for idx in cell_faces_idxes])
        cell_points = np.unique(np.concatenate(cell_faces))

        return {
            "faces": cell_faces_idxes,
            "points": cell_points,
            "neighbours": mesh.cell_neighbour[cell_idx],
        }

    @classmethod
    def _find_time_instants_subfolders(cls, path, fields_time_instants):
        full_path_with_label = lambda name: (name, os.path.join(path, name))

        def is_numeric(x):
            try:
                float(x)
                return True
            except ValueError:
                return False

        # if `fields_time_instants` is 'all_numeric', we load all the subfolders
        # of `path` whose name we can cast to a float. if 'first', we take only
        # the first one of those subfolders.
        if (
            fields_time_instants == "all_numeric"
            or fields_time_instants == "first"
        ):
            subfolders = next(os.walk(path))[1]
            subfolders = list(filter(is_numeric, subfolders))

            if len(subfolders) == 0:
                return None

            if fields_time_instants == "all_numeric":
                time_instant_subfolders = subfolders
            else:
                # we want a list in order to return an iterable
                time_instant_subfolders = [sorted(subfolders)[0]]
            return map(full_path_with_label, time_instant_subfolders)

        # if `fields_time_instants` is a list of strings, we take only the
        # subfolders whose name exactly matches with the strings in the list.
        elif isinstance(fields_time_instants, list):
            return map(full_path_with_label, fields_time_instants)
        else:
            raise ValueError(
                """Invalid value for the argument
                `fields_time_instants`"""
            )

    @classmethod
    def _find_fields_files(cls, fields_root_path, field_names):
        full_path_with_name = lambda name: (
            name,
            os.path.join(fields_root_path, name),
        )

        if field_names == "all":
            return map(
                full_path_with_name, next(os.walk(fields_root_path))[2]
            )
        elif isinstance(field_names, list):
            return map(full_path_with_name, field_names)
        else:
            raise ValueError("Invalid value for the argument `field_names`")

    @classmethod
    def _no_fail_boundary_field(cls, path):
        try:
            return Ofpp.parse_boundary_field(path)
        except IndexError:
            return None

    @classmethod
    def _no_fail_internal_field(cls, path):
        try:
            return Ofpp.parse_internal_field(path)
        except IndexError:
            return None

    @classmethod
    def _load_fields(cls, time_instant_path, field_names):
        field_files = cls._find_fields_files(time_instant_path, field_names)

        return dict(
            (name,
                (cls._no_fail_boundary_field(field_path),
                    cls._no_fail_internal_field(field_path)))
            for name, field_path in field_files
        )

    @classmethod
    def _build_time_instant_snapshot(cls, ofpp_mesh, time_instant_path,
        field_names):
        # TODO: e se la mesh cambia?
        return {
            "points": ofpp_mesh.points,
            "faces": np.array(ofpp_mesh.faces),
            "boundary": {
                key: cls._build_boundary(ofpp_mesh, key)
                for key in ofpp_mesh.boundary
            },
            "cells": {
                cell_id: cls._build_cells(ofpp_mesh, cell_id)
                for cell_id in range(len(ofpp_mesh.cell_faces))
            },
            "fields": cls._load_fields(time_instant_path, field_names)
        }

    @classmethod
    def read(cls, filename, fields_time_instants="first", field_names="all"):
        ofpp_mesh = Ofpp.FoamMesh(filename)

        time_instants = cls._find_time_instants_subfolders(filename,
            fields_time_instants)
        if time_instants is not None:
            return dict((name, cls._build_time_instant_snapshot(ofpp_mesh,
                path, field_names)) for name, path in time_instants)
        else:
            return cls._build_time_instant_snapshot(ofpp_mesh, filename,
                field_names)
