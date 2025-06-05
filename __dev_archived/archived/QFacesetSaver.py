# from localization import LStrings
# from xlib import qt as lib_qt

# from PyQt6.QtCore import *
# from PyQt6.QtGui import *
# from PyQt6.QtWidgets import *

# from .widgets.QSpinBoxCSWNumber import QSpinBoxCSWNumber
# from .widgets.QBackendPanel import QBackendPanel
# from .widgets.QPathEditCSWPaths import QPathEditCSWPaths
# from .widgets.QXPushButtonCSWSignal import QXPushButtonCSWSignal

# class QFacesetSaver(QBackendPanel):

#     def __init__(self, backend):
#         cs = self.cs = backend.get_control_sheet()
        
#         #self.q_error = QErrorSSDText(cs.error)
#         self.q_faceset_path = QPathEditCSWPaths(cs.faceset_path)
        
#         label_faces_count = lib_qt.QXLabel('Faces count')
#         q_faces_count = self.q_faces_count = QSpinBoxCSWNumber(cs.faces_count)
#         q_faces_count.reflect_state_to_widget(label_faces_count)
        
#         q_reload_signal = self.q_reload_signal = QXPushButtonCSWSignal(cs.reload_signal, text='Reload')
        
        
#         main_l = lib_qt.QXVBoxLayout()
#         #main_l.addWidget(self.q_error)
#         main_l.addWidget(self.q_faceset_path)
        
#         grid_l = lib_qt.QXGridLayout()
#         row = 0
        
#         fr = lib_qt.QXFrame(size_policy=(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed), layout=lib_qt.QXHBoxLayout([q_reload_signal]))
        
#         grid_l.addWidget(fr, row, 0, 1, 2, alignment=Qt.AlignmentFlag.AlignCenter)
#         row += 1
#         grid_l.addWidget(label_faces_count, row, 0, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter  )
#         grid_l.addWidget(q_faces_count, row, 1, alignment=Qt.AlignmentFlag.AlignLeft )
#         row += 1
        
#         main_l.addLayout(grid_l)
    
#         super().__init__(backend, 'Faceset saver',  main_l)
     