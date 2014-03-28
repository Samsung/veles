#!/usr/bin/python3.3 -O
"""
Created on November 7, 2013

@author: Lyubov Podoynitsina <lyubov.p@samsung.com>
"""


import glymur
import logging
import numpy
import os
from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QImage
import re
import sys


class MyWindow(QtGui.QWidget):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)

        self.resize(1175, 650)

        self.current_index = 0

        self.should_change_image = True

        self.last_action = "Nothing"
        self.max_cache = 0
        self.last_files = {}
        self.last_actions = []
        self.scaleFactor = 1.0
        self.current_label_size = [940, 540]
        self.current_image_size = [960, 540]
        self.should_stuck = False

        thumbs_path = "/data/veles/channels/usa_stb/new/images/old"
        thumbs_none = "thumbs_none"
        self.thumbs = []
        self.thumbs.append(os.path.join(thumbs_path, thumbs_none))
        fordel = []
        baddir = re.compile("bad", re.IGNORECASE)
        gooddir = re.compile("good", re.IGNORECASE)
        diffdir = re.compile("diff", re.IGNORECASE)
        deldir = re.compile("delete", re.IGNORECASE)
        wrongdir = re.compile("wrong", re.IGNORECASE)
        no_channeldir = re.compile("no channel", re.IGNORECASE)
        for root, dirs, files in os.walk(thumbs_path):
            for i, nme in enumerate(dirs):
                if (baddir.search(nme) is not None or
                        gooddir.search(nme) is not None or
                        diffdir.search(nme) is not None or
                        deldir.search(nme) is not None or
                        wrongdir.search(nme) is not None or
                        no_channeldir.search(nme) is not None):
                    fordel.append(i)
            while len(fordel) > 0:
                dirs.pop(fordel.pop())
            for fn in files:
                if fn.endswith(".JPEG") or fn.endswith(".png") or\
                        fn.endswith(".jpg") or fn.endswith(".bmp")\
                        or fn.endswith(".jp2"):
                    fullurl = os.path.join(root, fn)
                    self.thumbs.append(fullurl)
                    logging.info("Loading other format %s" % fn)
        self.thumbs.append(os.path.join(thumbs_path, thumbs_none))
        self.thumbs.sort()

        if len(self.thumbs) == 2:
            logging.info("Error: No image in folder")
        else:

            self.setWindowTitle('Sorting Images')
            self.setStyleSheet('background-color: rgb(255,249,225)')

            self.jpgs = {}
            for i in range(len(self.thumbs)):
                self.ind_path = self.thumbs[i].rfind("/")
                self.path_image = self.thumbs[i]
                dirnme = self.path_image[:self.ind_path]
                fnme = self.path_image[self.ind_path + 1:]
                if dirnme not in self.jpgs.keys():
                    self.jpgs[dirnme] = []
                self.jpgs[dirnme].append(fnme)
            keyList = self.jpgs.keys()
            sorted(keyList)

            self.button2 = QtGui.QPushButton(self)
            self.button2.setToolTip('Next image')
            self.button2.setText('Next')
            self.button2.setStyleSheet('font-size: 18pt; font-family: Courier;\
                            color: black; background-color: rgb(230,186,142)')
            self.button3 = QtGui.QPushButton(self)
            self.button3.setToolTip('Previous image')
            self.button3.setText('Previous')
            self.button3.setStyleSheet('font-size: 18pt; font-family: Courier;\
                            color: black; background-color: rgb(230,186,142)')
            self.button4 = QtGui.QPushButton(self)
            self.button4.setToolTip('Move current image to Good folder')
            self.button4.setText('Good logo')
            self.button4.setStyleSheet('font-size: 18pt; font-family: Courier;\
                            color: green; background-color: rgb(230,186,142)')
            self.button5 = QtGui.QPushButton(self)
            self.button5.setToolTip('Move current image to Bad folder')
            self.button5.setText('Bad lodo')
            self.button5.setStyleSheet('font-size: 18pt; font-family: Courier;\
                            color: red; background-color: rgb(230,186,142)')
            self.button6 = QtGui.QPushButton(self)
            self.button6.setToolTip('Move current image to Diff folder')
            self.button6.setText('Diff logo')
            self.button6.setStyleSheet('font-size: 18pt; font-family: Courier;\
                            color: black; background-color: rgb(230,186,142)')
            self.button7 = QtGui.QPushButton(self)
            self.button7.setToolTip('Move current image to No_channel folder')
            self.button7.setText('No channel')
            self.button7.setStyleSheet('font-size: 18pt; font-family: Courier;\
                            color: black; background-color: rgb(230,186,142)')
            self.button8 = QtGui.QPushButton(self)
            self.button8.setToolTip('Cancel last move of image')
            self.button8.setText('Cancel')
            self.button8.setStyleSheet('font-size: 18pt; font-family: Courier;\
                            color: red; background-color: rgb(230,186,142)')

            self.button9 = QtGui.QPushButton(self)
            self.button9.setText('Zoom in')
            self.button9.setToolTip('Zoom in current image')
            self.button9.setStyleSheet('font-size: 18pt; font-family: Courier;\
                            color: black; background-color: rgb(230,186,142)')

            self.button10 = QtGui.QPushButton(self)
            self.button10.setText('Zoom out')
            self.button10.setToolTip('Zoom out current image')
            self.button10.setStyleSheet('font-size: 18pt; font-family:\
                                        Courier; color: black;\
                                        background-color: rgb(230,186,142)')
            self.button11 = QtGui.QPushButton(self)
            self.button11.setText('Zoom fix')
            self.button11.setToolTip('Fix a previous zoom')
            self.button11.setStyleSheet('font-size: 18pt; font-family:\
                                        Courier; color: black;\
                                        background-color: rgb(230,186,142)')
            self.button12 = QtGui.QPushButton(self)
            self.button12.setText('Wrong channel')
            self.button12.setToolTip('Number of channel don t match\
                                     to the logo')
            self.button12.setStyleSheet('font-size: 18pt; font-family:\
                                        Courier; color: black;\
                                        background-color: rgb(230,186,142)')

            self.button13 = QtGui.QPushButton(self)
            self.button13.setText('Delete')
            self.button13.setStyleSheet('font-size: 18pt; font-family:\
                                        Courier; color: black;\
                                        background-color: rgb(230,186,142)')

            self.hor_layout = QtGui.QHBoxLayout(self)
            self.ver_layout3 = QtGui.QVBoxLayout(self)
            self.grid_layout = QtGui.QGridLayout(self)
            self.label2 = QtGui.QLabel(self)
            self.hor_layout2 = QtGui.QHBoxLayout(self)
            self.hor_layout2.addWidget(self.button3)
            self.hor_layout2.addWidget(self.button2)
            self.ver_layout2 = QtGui.QVBoxLayout(self)

            self.comboBox = QtGui.QComboBox(self)
            self.comboBox.addItems(list(self.jpgs.keys()))
            self.comboBox.currentIndexChanged[str].connect(self.changeDir)
            self.ver_layout2.addWidget(self.comboBox)
            self.comboBox2 = QtGui.QComboBox(self)

            ind_path = self.thumbs[1].rfind("/")
            path_image = self.thumbs[1]
            path_channel = path_image[:ind_path]
            self.changeDir(path_channel)
            self.ver_layout2.addWidget(self.comboBox2)
            self.ver_layout3.addLayout(self.hor_layout2)
            self.hor_layout2.addLayout(self.ver_layout2)
            self.grid_layout.addLayout(self.ver_layout3, 1, 0, 1, 3)
            self.ver_layout3.addWidget(self.label2)

            self.label = QtGui.QLabel(self)
            self.label.resize(940, 540)
            self.scrollArea = QtGui.QScrollArea()
            self.scrollArea.setWidget(self.label)
            self.scrollArea.setWidgetResizable(True)
            self.scrollArea.setAlignment
            self.grid_layout.addWidget(self.scrollArea, 0, 0, 1, 1)

            self.ver_layout = QtGui.QVBoxLayout(self)
            self.ver_layout.addWidget(self.button4)
            self.ver_layout.addWidget(self.button5)
            self.ver_layout.addWidget(self.button6)
            self.ver_layout.addWidget(self.button7)
            self.ver_layout.addWidget(self.button12)
            self.ver_layout.addWidget(self.button13)
            self.ver_layout.addWidget(self.button8)
            self.ver_layout.addWidget(self.button9)
            self.ver_layout.addWidget(self.button10)
            self.ver_layout.addWidget(self.button11)

            self.grid_layout.addLayout(self.ver_layout, 0, 2, 1, 1)
            self.ver_spacer = QtGui.QSpacerItem(20, 40)
            self.grid_layout.addItem(self.ver_spacer, 0, 1, 1, 1)

            self.hor_layout.addLayout(self.grid_layout)

            self.button2.clicked.connect(lambda: self.nextImage(self.thumbs))
            self.button3.clicked.connect(lambda:
                                         self.previousImage(self.thumbs))
            self.button4.clicked.connect(lambda: self.goodImage(self.thumbs))
            self.button5.clicked.connect(lambda: self.badImage(self.thumbs))
            self.button6.clicked.connect(lambda: self.diffImage(self.thumbs))
            self.button7.clicked.connect(lambda:
                                         self.no_channelImage(self.thumbs))
            self.button4.clicked.connect(lambda: self.PassLeft())
            self.button5.clicked.connect(lambda: self.PassLeft())
            self.button6.clicked.connect(lambda: self.PassLeft())
            self.button7.clicked.connect(lambda: self.PassLeft())
            self.button8.clicked.connect(lambda:
                                         self.cancelLastMove(self.last_action))
            self.button9.clicked.connect(lambda: self.scaleImage(1.25))
            self.button10.clicked.connect(lambda: self.scaleImage(0.8))
            self.button11.clicked.connect(lambda: self.stuck())
            self.button12.clicked.connect(lambda:
                                          self.wrongImage(self.thumbs))
            self.button13.clicked.connect(lambda:
                                          self.deleteImage(self.thumbs))
            self.button12.clicked.connect(lambda: self.PassLeft())
            self.button13.clicked.connect(lambda: self.PassLeft())

            self.total_images = len(self.thumbs)
            self.passed_images = 0
            self.left_images = self.total_images
            self.nextImage(self.thumbs)

    def stuck(self):
        if not self.should_stuck:
            self.should_stuck = True
        else:
            self.should_stuck = False

    def scaleImage(self, factor):
        self.scaleFactor *= factor
        self.label.resize(self.scaleFactor * 940, self.scaleFactor * 540)
        self.current_image = self.current_image.scaled(self.scaleFactor * 960,
                                                       self.scaleFactor * 540)
        self.label.setPixmap(self.current_image)
        self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), factor)
        self.adjustScrollBar(self.scrollArea.verticalScrollBar(), factor)

    def normalSize(self, current_image):
        self.label.resize(940, 540)
        self.current_image = self.current_image.scaled(960, 540)
        self.label.setPixmap(self.current_image)
        self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), 1)
        self.adjustScrollBar(self.scrollArea.verticalScrollBar(), 1)

    def from_jp2(self, fnme):
        j2 = glymur.Jp2k(fnme)
        a2 = j2.read()
        if j2.box[2].box[1].colorspace != 16:
            return self.yuv_to_rgb(a2)
        return a2

    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(int(factor * scrollBar.value()
                               + ((factor - 1) * scrollBar.pageStep() / 2)))

    def numpyToQImage(self, path_name):
        bgra = self.from_jp2(path_name)
        h, w, = bgra.shape[:2]
        result = QImage(bgra.data, w, h, QImage.Format_RGB888)
        return result

    def yuv_to_rgb(self, a):
        aa = a.astype(numpy.float32)
        aaa = numpy.empty_like(aa)
        aaa[:, :, 0:1] = aa[:, :, 0:1] + (aa[:, :, 2:3] - 128) * 1.402
        aaa[:, :, 1:2] = (aa[:, :, 0:1] + (aa[:, :, 1:2] - 128) * (-0.34414) +
                          (aa[:, :, 2:3] - 128) * (-0.71414))
        aaa[:, :, 2:3] = aa[:, :, 0:1] + (aa[:, :, 1:2] - 128) * 1.772
        numpy.clip(aaa, 0.0, 255.0, aa)
        return aa.astype(numpy.uint8)

    def PassLeft(self):
        self.label2.setText("already sorted %s/left to sort %s"
                            % (self.passed_images, self.left_images - 2))
        self.label2.setStyleSheet('font-size: 12pt; font-family: Courier;')
        self.label2.setFixedHeight(20)

    def keyPressEvent(self, event):

        if event.key() == QtCore.Qt.Key_6:
            self.nextImage(self.thumbs)
            return
        if event.key() == QtCore.Qt.Key_4:
            self.previousImage(self.thumbs)
            return
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()
        logging.info("KeyPressEvent %s" % self.current_index)

    def setImage(self, file_name):
        if (file_name.endswith(".JPEG") or file_name.endswith(".png") or
                file_name.endswith(".jpg") or file_name.endswith(".bmp")):
            pixmap = QtGui.QPixmap(self.path_to_current_image)
            if not self.should_stuck:
                self.scaleFactor = 1
                pixmap = pixmap.scaled(960, 540, QtCore.Qt.KeepAspectRatio)
                self.current_image = pixmap
                self.label.setPixmap(pixmap)
                self.normalSize(self.current_image)
            else:
                self.current_image = pixmap
                self.label.setPixmap(pixmap)
        if file_name.endswith(".jp2"):
            img = self.numpyToQImage(self.path_to_current_image)
            pixmap = QtGui.QPixmap.fromImage(img)
            if not self.should_stuck:
                self.scaleFactor = 1
                pixmap = pixmap.scaled(960, 540, QtCore.Qt.KeepAspectRatio)
                self.current_image = pixmap
                self.label.setPixmap(pixmap)
                self.normalSize(self.current_image)
            else:
                self.current_image = pixmap
                self.label.setPixmap(pixmap)
        logging.info("Change image %s" % self.current_index)

    def changeImage(self, file_name):
        if self.comboBox2.currentIndexChanged[str]:
            self.should_change_image = True
        if self.should_change_image:
            self.path_to_current_image = os.path.join(self.path_to_current_dir,
                                                      file_name)
            for i in range(len(self.thumbs)):
                if self.thumbs[i] == self.path_to_current_image:
                    self.current_index = i
            self.setImage(file_name)

    def changeDir(self, pathToImage):
        self.should_change_image = True
        self.comboBox2.clear()
        self.comboBox2.addItems(self.jpgs[pathToImage])
        self.comboBox2.currentIndexChanged[str].connect(self.changeImage)
        self.path_to_current_dir = pathToImage
        self.index = self.comboBox2.currentIndex()

    def nextImage(self, thumbs):
        try:
            ind_path = thumbs[self.current_index].rfind("/")
            self.path_to_current_image = thumbs[self.current_index]
            file_name = self.path_to_current_image[ind_path + 1:]
            #self.scaleFactor = 1
            self.current_index += 1
            self.setImage(file_name)
            self.index = self.comboBox2.currentIndex()
            self.index += 1
            self.should_change_image = False
            self.comboBox2.setCurrentIndex(self.index)
            logging.info("Next Image %s" % self.current_index)
        except:
            pass

    def previousImage(self, thumbs):
        try:
            ind_path = thumbs[self.current_index].rfind("/")
            self.path_to_current_image = thumbs[self.current_index]
            file_name = self.path_to_current_image[ind_path + 1:]
            #self.scaleFactor = 1
            self.current_index -= 1
            self.setImage(file_name)
            self.index = self.comboBox2.currentIndex()
            self.index -= 1
            self.should_change_image = False
            self.comboBox2.setCurrentIndex(self.index)
            logging.info("Previous Image %s" % self.current_index)
        except:
            pass

    def moveImage(self, thumbs, name_dir):
        if self.left_images > 2:
            self.should_append = True
            ind_path = thumbs[self.current_index].rfind("/")
            path_image = thumbs[self.current_index]
            path_channel = path_image[:ind_path]
            current_file = path_image[ind_path + 1:]
            path_channel_dir = os.path.join(path_channel, name_dir)
            try:
                os.mkdir(path_channel_dir)
            except OSError:
                pass
            os.rename(path_image, os.path.join(path_channel_dir, current_file))
            self.last_move_file = os.path.join(path_channel_dir, current_file)
            while self.max_cache < 10 and self.should_append:
                if name_dir not in self.last_files.keys():
                    self.last_files[name_dir] = []
                self.last_files[name_dir].append(self.last_move_file)
                self.max_cache += 1
                self.should_append = False
                self.last_actions.append(name_dir)
            self.index = self.comboBox2.currentIndex()
            self.should_change_image = False
            self.comboBox2.removeItem(self.index)
            path_image = thumbs[self.current_index]
            current_file = path_image[(path_image.rfind("/") + 1):]
            self.path_to_current_image = thumbs[self.current_index]
            self.setImage(current_file)
            logging.info("%s image %s" % (name_dir, self.current_index))
            self.passed_images += 1
            self.left_images -= 1
            self.last_action = "%s image" % (name_dir)
        if self.left_images == 2:
            self.label.setText("End of sorting, good job!")
            self.label.setStyleSheet('font-size: 46pt; font-family: Courier;')

    def goodImage(self, thumbs):
        self.moveImage(thumbs, "Good")

    def wrongImage(self, thumbs):
        self.moveImage(thumbs, "Wrong")

    def badImage(self, thumbs):
        self.moveImage(thumbs, "Bad")

    def diffImage(self, thumbs):
        self.moveImage(thumbs, "Diff")

    def no_channelImage(self, thumbs):
        self.moveImage(thumbs, "No channel")

    def deleteImage(self, thumbs):
        self.moveImage(thumbs, "Delete")

    def moveImageBack(self, dirnme):
        current_dir = self.last_files[dirnme]
        self.last_move_file = current_dir[-1]
        ind_path = self.last_move_file.rfind("/")
        path_image = self.last_move_file
        path_channel_dir = path_image[:ind_path]
        current_file = path_image[ind_path + 1:]
        ind_path_ch = path_channel_dir.rfind("/")
        path_channel = path_image[:ind_path_ch]
        os.rename(path_image, os.path.join(path_channel, current_file))
        self.should_change_image = False
        self.comboBox2.addItem("%s" % (current_file))
        self.path_to_current_image = os.path.join(path_channel, current_file)
        self.setImage(current_file)
        logging.info("Cancel move to %s Image %s" % (dirnme,
                                                     self.current_index))
        self.passed_images -= 1
        self.left_images += 1
        self.last_files[dirnme].remove(self.last_move_file)
        self.last_actions.remove(self.last_actions[len(self.last_actions) - 1])

    def cancelLastMove(self, last_action):
        if last_action == "Nothing":
            pass
        if last_action == "Next image":
            self.previousImage(self.thumbs)
        if last_action == "Previous image":
            self.nextImage(self.thumbs)
        if len(self.last_actions):
            self.moveImageBack(self.last_actions[-1])


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    app.setApplicationName('MyWindow')
    #if __debug__:
    #    logging.basicConfig(level=logging.DEBUG)
    #else:
    logging.basicConfig(level=logging.INFO)

    main = MyWindow()
    main.show()
    logging.info("End of job")

    sys.exit(app.exec_())
