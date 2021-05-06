"""
Abstract class to specialize for `vtk` handlers.

It contains the recurrent functions shared between the different file handlers.
"""
from abc import ABC

class BaseVTKHandler(ABC):

    from vtk import vtkPoints, vtkCellArray

    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

    _vtk_to_numpy_ = vtk_to_numpy
    _numpy_to_vtk_ = numpy_to_vtk
    _points_ = vtkPoints
    _cells_ = vtkCellArray

    @classmethod
    def _read_point_data(cls, vtkdata):
        """
        Extract the point data

        :param vtkObject vtkdata: a vtk object (vtkPolyData, vtkUnstructuredGrid, ...)
        """
        result = {}
        for i in range(vtkdata.GetPointData().GetNumberOfArrays()):
            array = cls._vtk_to_numpy_(vtkdata.GetPointData().GetArray(i))
            name = vtkdata.GetPointData().GetArrayName(i)
            result[name] = array
        return result

    @classmethod
    def _read_cell_data(cls, vtkdata):
        """
        Extract the cell data

        :param vtkObject vtkdata: a vtk object (vtkPolyData, vtkUnstructuredGrid, ...)
        """
        result = {}
        for i in range(vtkdata.GetCellData().GetNumberOfArrays()):
            array = cls._vtk_to_numpy_(vtkdata.GetCellData().GetArray(i))
            name = vtkdata.GetCellData().GetArrayName(i)
            result[name] = array
        return result

    @classmethod
    def _write_point_data(cls, vtkdata, data):
        """
        Write all the arrays in `data['point_data']` to the `vtkdata`.

        :param vtkObject vtkdata: a vtk object (vtkPolyData, vtkUnstructuredGrid, ...)
        :param dict data: dictionary 
        """
        for name, array in data['point_data'].items():
            vtkarray = cls._numpy_to_vtk_(array)
            vtkarray.SetName(name)
            vtkdata.GetPointData().AddArray(vtkarray)

    @classmethod
    def _write_cell_data(cls, vtkdata, data):
        """
        Write all the arrays in `data['cell_data']` to the `vtkdata`.

        :param vtkObject vtkdata: a vtk object (vtkPolyData, vtkUnstructuredGrid, ...)
        :param dict data: dictionary 
        """
        for name, array in data['cell_data'].items():
            vtkarray = cls._numpy_to_vtk_(array)
            vtkarray.SetName(name)
            vtkdata.GetCellData().AddArray(vtkarray)
