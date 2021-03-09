from .vtkhandler import VTKHandler

class VTPHandler(VTKHandler):

    from vtk import vtkXMLPolyDataReader, vtkXMLPolyDataWriter

    _reader_ = vtkXMLPolyDataReader
    _writer_ = vtkXMLPolyDataWriter
