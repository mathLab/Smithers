from .basevtkhandler import BaseVTKHandler


class VTKHandler(BaseVTKHandler):
    """
    Handler for .VTK files.
    """
    from vtk import vtkPolyData, vtkUnstructuredGrid
    from vtk import vtkPolyDataReader, vtkPolyDataWriter
    from vtk import vtkUnstructuredGridReader, vtkUnstructuredGridWriter
    from vtk import vtkXMLPolyDataReader, vtkXMLPolyDataWriter

    vtk_format = {
        'polydata': {'writer': vtkPolyDataWriter,
                     'reader': vtkPolyDataReader,
                     'type':   vtkPolyData},

        'unstructured': {'writer': vtkUnstructuredGridWriter,
                         'reader': vtkUnstructuredGridReader,
                         'type':   vtkUnstructuredGrid},

        'xml_polydata': {'writer': vtkXMLPolyDataWriter,
                         'reader': vtkXMLPolyDataReader,
                         'type':   vtkPolyData},
    }

    @classmethod
    def read(cls, filename, fmt='polydata'):

        if fmt not in cls.vtk_format.keys():
            raise ValueError('`fmt` is invalid')

        reader = cls.vtk_format[fmt]['reader']()
        reader.SetFileName(filename)
        reader.Update()
        data_dict = cls.vtk2dict(reader.GetOutput())
        # data_dict['format'] = format # TODO: format attribute?
        return data_dict

    @classmethod
    def vtk2dict(cls, data):
        result = {'cells': [], 'points': None}

        for id_cell in range(data.GetNumberOfCells()):
            cell = data.GetCell(id_cell)
            print(cell)
            result['cells'].append([
                cell.GetPointId(id_point)
                for id_point in range(cell.GetNumberOfPoints())
            ])

        result['points'] = cls._vtk_to_numpy_(data.GetPoints().GetData())

        result['point_data'] = cls._read_point_data(data)
        result['cell_data'] = cls._read_cell_data(data)

        return result

    @classmethod
    def dict2vtk(cls, data, fmt):
        """ TODO """

        vtkdata = cls.vtk_format[fmt]['type']()

        vtk_points = cls._points_()
        vtk_points.SetData(cls._numpy_to_vtk_(data['points']))

        vtk_cells = cls._cells_()
        for cell in data['cells']:
            vtk_cells.InsertNextCell(len(cell), cell)

        cls._write_point_data(vtkdata, data)
        cls._write_cell_data(vtkdata, data)

        vtkdata.SetPoints(vtk_points)
        vtkdata.SetPolys(vtk_cells)

        return vtkdata

    @classmethod
    def write(cls, filename, data, fmt='polydata'):

        if fmt not in cls.vtk_format.keys():
            raise ValueError('`fmt` is invalid')

        vtkdata = cls.dict2vtk(data, fmt)
        writer = cls.vtk_format[fmt]['writer']()
        writer.SetFileName(filename)
        writer.SetInputData(vtkdata)
        writer.Write()
