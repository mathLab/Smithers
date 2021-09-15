from .basevtkhandler import BaseVTKHandler

class VTKHandler(BaseVTKHandler):
    """
    Handler for .VTK files.
    """
    from vtk import vtkPolyDataReader, vtkPolyDataWriter
    from vtk import vtkUnstructuredGridReader, vtkUnstructuredGridWriter
    from vtk import vtkPolyData, vtkUnstructuredGrid

    _data_type_ = vtkPolyData

    _reader_ = vtkUnstructuredGridReader
    _writer_ = vtkUnstructuredGridWriter

    @classmethod
    def read(cls, filename):
        reader = cls._reader_()
        reader.SetFileName(filename)
        reader.Update()
        return parse(reader.GetOutput())

    @classmethod
    def parse(cls, data):
        result = {'cells': [], 'points': None}

        for id_cell in range(data.GetNumberOfCells()):
            cell = data.GetCell(id_cell)
            result['cells'].append([
                cell.GetPointId(id_point)
                for id_point in range(cell.GetNumberOfPoints())
            ])

        result['points'] = cls._vtk_to_numpy_(data.GetPoints().GetData())

        result['point_data'] = cls._read_point_data(data)
        result['cell_data'] = cls._read_cell_data(data)

        return result

    @classmethod
    def write(cls, filename, data):
        """ TODO """

        from vtk import vtkPolyData, vtkPoints, vtkCellArray
        from vtk.util.numpy_support import numpy_to_vtk

        polydata = vtkPolyData()

        vtk_points = vtkPoints()
        vtk_points.SetData(numpy_to_vtk(data['points']))

        vtk_cells = vtkCellArray()
        for cell in data['cells']:
            vtk_cells.InsertNextCell(len(cell), cell)

        cls._write_point_data(polydata, data)
        cls._write_cell_data(polydata, data)

        polydata.SetPoints(vtk_points)
        polydata.SetPolys(vtk_cells)

        writer = cls._writer_()
        writer.SetFileName(filename)
        writer.SetInputData(polydata)
        writer.Write()
