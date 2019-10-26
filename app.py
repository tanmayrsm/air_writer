import sys

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QFrame, QWidget
from PyQt5.QtWidgets import QGridLayout, QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QDesktopWidget, QLabel, QPushButton

from camera import VideoStream
from Pipeline import Pipeline


class MainGUI(QWidget):
    
    def __init__(self):
        super().__init__()
        self.init_pipeline()
        self.init_UI()
        
        return
    
    def init_pipeline(self):
        self.pipeline = Pipeline()
        self.engine = 'EN'
        
        return
    
    def init_UI(self):
        self.setGeometry(0, 0, 0, 0)
        self.setStyleSheet('QWidget {background-color: #ffffff;}')
        self.setWindowIcon(QIcon('assets/logo.png'))
        self.setWindowTitle('Air-Writing Data Generator')
        
        self.btn_conn = QPushButton('Connect Camera')
        self.btn_conn.setMinimumSize(500, 40)
        self.btn_conn_style_0 = 'QPushButton {background-color: #00a86c; border: none; color: #ffffff; font-family: ubuntu, arial; font-size: 16px;}'
        self.btn_conn_style_1 = 'QPushButton {background-color: #ff6464; border: none; color: #ffffff; font-family: ubuntu, arial; font-size: 16px;}'
        self.btn_conn.setStyleSheet(self.btn_conn_style_0)
        
        self.cam_feed = QLabel()
        self.cam_feed.setMinimumSize(640, 480)
        self.cam_feed.setAlignment(Qt.AlignCenter)
        self.cam_feed.setFrameStyle(QFrame.StyledPanel)
        self.cam_feed.setStyleSheet('QLabel {background-color: #000000;}')
        
        h_box1 = QHBoxLayout()
        h_box1.addWidget(self.btn_conn)
        
        v_box1 = QVBoxLayout()
        v_box1.addLayout(h_box1)

        v_box1.addWidget(self.cam_feed)
        
        g_box0 = QGridLayout()
        g_box0.addLayout(v_box1, 0, 0, -1, 2)
        
        self.setLayout(g_box0)
        
        self.flg_conn = False
        
        self.btn_conn.clicked.connect(self.connect)
        
        return
    
    def moveWindowToCenter(self):
        window_rect = self.frameGeometry()
        screen_cent = QDesktopWidget().availableGeometry().center()
        window_rect.moveCenter(screen_cent)
        self.move(window_rect.topLeft())
        
        return
    
    def connect(self):
        self.flg_conn = not self.flg_conn
        if self.flg_conn:
            self.btn_conn.setStyleSheet(self.btn_conn_style_1)
            self.btn_conn.setText('Disconnect Camera')
            self.video = VideoStream()
            self.timer = QTimer()
            self.timer.timeout.connect(self.update)
            self.timer.start(50)
        else:
            self.btn_conn.setStyleSheet(self.btn_conn_style_0)
            self.btn_conn.setText('Connect Camera')
            self.cam_feed.clear()
            self.timer.stop()
            self.video.clear()
        
        return
    
    def update(self):
        frame = self.video.getFrame(flip=1)
        if not frame is None:
            frame = self.pipeline.run_inference(frame, self.engine, True)
            frame = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
            self.cam_feed.setPixmap(QPixmap.fromImage(frame))
        else:
            self.cam_feed.clear()
        
        return

    def closeEvent(self, event):
        if self.flg_conn:
            self.connect()
        
        return

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    gui = MainGUI()
    gui.show()
    gui.setFixedSize(gui.size())
    gui.moveWindowToCenter()
    sys.exit(app.exec_())
