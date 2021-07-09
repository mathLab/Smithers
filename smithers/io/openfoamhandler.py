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
    def _build_boundary(cls, mesh, boundary_name):
        """Extract information about a boundary.

        :param mesh: An Ofpp OpenFOAM mesh.
        :type mesh: Ofpp.mesh_parser.FoamMesh
        :param boundary_name: The boundary name (must be a key of the dictionary
            `mesh.boundary`).
        :type boundary_name: str
        :returns: A dictionary which contains the keys 'faces', 'points',
            'type'.
        :rtype: dict
        """
        data = mesh.boundary[boundary_name]

        # extract faces which compose the boundary
        bd_faces_indexes = np.array(list(range(data.start, data.start + data.num)))
        bd_faces = np.concatenate([mesh.faces[idx] for idx in bd_faces_indexes])

        # extract a list of unique points which compose the boundary
        bd_points = np.unique(bd_faces)

        return {"faces": bd_faces_indexes, "points": bd_points, "type": data.type}

    @classmethod
    def _build_cells(cls, mesh, cell_idx):
        """Extract information about a cell.

        :param mesh: An Ofpp OpenFOAM mesh.
        :type mesh: Ofpp.mesh_parser.FoamMesh
        :param cell_idx: The index of the cell in the list `mesh.cell_faces`.
        :type cell_idx: int
        :returns: A dictionary which contains the keys 'faces', 'points',
            'neighbours'.
        :rtype: dict
        """
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
        """Finds all the time instants in the subfolders of `path` (at the moment
        this is only used to find the time evolution of fields).

        :param path: The base folder for the mesh.
        :type path: str
        :param fields_time_instants: One of `'all_numeric'` (select all the
            subfolders of `path` whose name can be converted to float),
            `'first'` (same of `'all_numeric'`, but return only the folder
            with the lowest name), or a `list` of exact names of the selected
            subfolders.
        :type fields_time_instants: str or list
        :returns: A list of tuples (first item: subfolder name, second item:
            subfolder full path).
        :rtype: list
        """
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
            return list(map(full_path_with_label, fields_time_instants))
        else:
            raise ValueError(
                """Invalid value for the argument
                `fields_time_instants`"""
            )

    @classmethod
    def _find_fields_files(cls, fields_root_path, field_names):
        """Finds all the fields in the subfolders of `fields_root_path`.

        :param fields_root_path: The base folder for the time instants at which
            we are looking for the fields.
        :type fields_root_path: str
        :param field_names: Refer to the documentation of the parameter
            `field_names` for the function :func:`_load_fields`.
        :type field_names: str or list
        :returns: A list of tuples (first item: subfolder name, second item:
            subfolder full path).
        :rtype: list
        """
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
        """Parse the boundary field at the given `path`.

        :param path: The path which contains the boundary field.
        :type path: str
        :returns: The boundary field, or `None` if the parse fails.
        :rtype: list
        """
        try:
            return Ofpp.parse_boundary_field(path)
        except IndexError:
            return None

    @classmethod
    def _no_fail_internal_field(cls, path):
        """Parse the internal field at the given `path`.

        :param path: The path which contains the internal field.
        :type path: str
        :returns: The internal field, or `None` if the parse fails.
        :rtype: list
        """
        try:
            return Ofpp.parse_internal_field(path)
        except IndexError:
            return None

    @classmethod
    def _load_fields(cls, time_instant_path, field_names):
        """Read all the fields at the given path.

        :param time_instant_path: The base folder of the time instant at which
            we consider the fields.
        :type time_instant_path: str
        :param field_names: The string `'all'` (select all the subfolders of
            `fields_root_path`) or a `list` of exact names of the selected
            subfolders.
        :type field_names: str or list
        :returns: A dictionary of fields, whose names are the names of the
            fields, and the values are 2-tuple (first item: the boundary field,
            second item: the internal field).
        :rtype: dict
        """
        field_files = cls._find_fields_files(time_instant_path, field_names)

        return dict(
            (name,
                (cls._no_fail_boundary_field(field_path),
                    cls._no_fail_internal_field(field_path)))
            for name, field_path in field_files
        )

    @classmethod
    def _build_time_instant_snapshot(cls, mesh, time_instant_path,
        field_names):
        """Read all the content available for the time instant at the given
            path.

        :param mesh: An Ofpp OpenFOAM mesh.
        :type mesh: Ofpp.mesh_parser.FoamMesh
        :param time_instant_path: The base folder of the time instant.
        :type time_instant_path: str
        :param field_names: Refer to the documentation of the parameter
            `field_names` for the function :func:`_load_fields`.
        :type field_names: str or list
        :returns: A dictionary with keys:
            * `'points'`: points of the mesh at the given time instants;
            * `'faces'`: faces of the mesh at the given time instants;
            * `'boundary'`: output of :func:`_build_boundary`;
            * `'cells'`: cells of the mesh at the given time instants;
            * `'fields'`: output of :func:`_load_fields`;
        :rtype: dict
        """

        # TODO: e se la mesh cambia?
        return {
            "points": mesh.points,
            "faces": np.array(mesh.faces),
            "boundary": {
                key: cls._build_boundary(mesh, key)
                for key in mesh.boundary
            },
            "cells": {
                cell_id: cls._build_cells(mesh, cell_id)
                for cell_id in range(len(mesh.cell_faces))
            },
            "fields": cls._load_fields(time_instant_path, field_names)
        }

    @classmethod
    def read(cls, filename, fields_time_instants="first", field_names="all"):
        """Read the OpenFOAM mesh at the given path. Parsing of multiple time
        instants and of fields is supported. At the moment the mesh isn not
        allowed to change (WIP).

        :param filename: The root folder of the mesh.
        :type filename: str
        :param fields_time_instants: Refer to the documentation of the parameter
            `field_names` for the function
            :func:`_find_time_instants_subfolders`.
        :type fields_time_instants: str or list
        :param field_names: Refer to the documentation of the parameter
            `field_names` for the function :func:`_load_fields`.
        :type field_names: str or list
        :returns: A dictionary whose keys are the time instants found for this
            mesh, and the values are the corresponding output of
            :func:`_build_time_instant_snapshot`. If only one time isntant is
            found the upper dictionary is skipped (the output of the function
            is the output of :func:`_build_time_instant_snapshot`).
        :rtype: dict
        """

        ofpp_mesh = Ofpp.FoamMesh(filename)

        time_instants = cls._find_time_instants_subfolders(filename,
            fields_time_instants)
        if time_instants is not None:
            return dict((name, cls._build_time_instant_snapshot(ofpp_mesh,
                path, field_names)) for name, path in time_instants)
        else:
            return cls._build_time_instant_snapshot(ofpp_mesh, filename,
                field_names)
