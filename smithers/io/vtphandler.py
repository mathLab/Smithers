from .vtkhandler import VTKHandler


class VTPHandler(VTKHandler):

    cls_format = 'xml_polydata'

    @classmethod
    def read(cls, filename):
        return super().read(filename, fmt=cls.cls_format)

    @classmethod
    def write(cls, filename, data):
        super().write(filename, data, fmt=cls.cls_format)
