from .basevtkhandler import BaseVTKHandler


class STLHandler(BaseVTKHandler):
    """
    Handler for .STL files.
    """
    from vtk import vtkSTLReader, vtkSTLWriter
    from vtk import vtkPolyData

    _data_type_ = vtkPolyData

    _reader_ = vtkSTLReader
    _writer_ = vtkSTLWriter

    @classmethod
    def read(cls, filename):

        reader = cls._reader_()
        reader.SetFileName(filename)
        reader.Update()
        data = reader.GetOutput()
        result = {'cells': [], 'points': None}

        for id_cell in range(data.GetNumberOfCells()):
            cell = data.GetCell(id_cell)
            result['cells'].append([
                cell.GetPointId(id_point)
                for id_point in range(cell.GetNumberOfPoints())
            ])

        result['points'] = cls._vtk_to_numpy_(data.GetPoints().GetData())

        return result

    @classmethod
    def write(cls, filename, data):
        """ TODO """

        from vtk import vtkPoints, vtkCellArray
        from vtk.util.numpy_support import numpy_to_vtk

        polydata = cls._data_type_()

        vtk_points = vtkPoints()
        vtk_points.SetData(numpy_to_vtk(data['points']))

        vtk_cells = vtkCellArray()
        for cell in data['cells']:
            vtk_cells.InsertNextCell(len(cell), cell)

        polydata.SetPoints(vtk_points)
        polydata.SetPolys(vtk_cells)

        writer = cls._writer_()
        writer.SetFileName(filename)
        writer.SetInputData(polydata)
        writer.Write()
