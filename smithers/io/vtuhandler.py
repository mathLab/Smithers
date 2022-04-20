"""
    Module for .VTU files

    .VTU files are VTK files with XML syntax containing vtkUnstructuredGrid.
    Further information related with the file format available at url:
    https://www.vtk.org/VTK/img/file-formats.pdf
"""

from .basevtkhandler import BaseVTKHandler


class VTUHandler(BaseVTKHandler):
    """
    Handler for .VTU files.
    """
    from vtk import vtkXMLUnstructuredGridReader, vtkXMLUnstructuredGridWriter
    from vtk import vtkUnstructuredGrid
    from vtk import VTK_POLYGON

    _data_type_ = vtkUnstructuredGrid

    _reader_ = vtkXMLUnstructuredGridReader
    _writer_ = vtkXMLUnstructuredGridWriter

    @classmethod
    def read(cls, filename):
        """
        """
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

        result['point_data'] = cls._read_point_data(data)
        result['cell_data'] = cls._read_cell_data(data)

        return result

    @classmethod
    def write(cls, vtuOrg, filename, data):
        """
        Method to save the dataset to `filename`. The dataset `data` should be
        a dictionary containing the requested information. The obtained
        `filename` is a well-formatted VTU file.

        :param vtuOrg : original vtu to specify the cell types
        :param str filename: the name of the file to write.
        :param dict data: the dataset to save.

        .. note::
          The cells will be saved with the same format as they were,
          no need to specify the cell type.
        """
        #VTK object
        reader = cls._reader_()
        reader.SetFileName(vtuOrg)
        reader.Update()
        vtkObj = reader.GetOutput()

        #define paramaters needed for setCell
        CellTypes=vtkObj.GetCellTypesArray()
        faceLoc=vtkObj.GetFaceLocations()
        face=vtkObj.GetFaces()

        unstructured_grid = cls._data_type_()

        points = cls._points_()
        points.SetData(cls._numpy_to_vtk_(data['points']))

        cells = cls._cells_()
        for cell in data['cells']:
            cells.InsertNextCell(len(cell), cell)

        cls._write_point_data(unstructured_grid, data)
        cls._write_cell_data(unstructured_grid, data)

        unstructured_grid.SetPoints(points)
        unstructured_grid.SetCells(CellTypes, cells, faceLoc, face)

        writer = cls._writer_()
        writer.SetFileName(filename)
        writer.SetInputData(unstructured_grid)
        writer.Write()
