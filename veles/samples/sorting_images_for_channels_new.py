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

        self.resize(1590, 650)

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
        cnndir = re.compile("cnn", re.IGNORECASE)
        abcdir = re.compile("abc", re.IGNORECASE)
        histdir = re.compile("hist", re.IGNORECASE)
        usadir = re.compile("usa", re.IGNORECASE)
        knbcdir = re.compile("knbc", re.IGNORECASE)
        unidir = re.compile("uni", re.IGNORECASE)
        cbsdir = re.compile("cbs", re.IGNORECASE)
        tntdir = re.compile("tnt", re.IGNORECASE)
        espndir = re.compile("espn", re.IGNORECASE)
        cnbcdir = re.compile("cnbc", re.IGNORECASE)
        msnbcdir = re.compile("msnbc", re.IGNORECASE)
        weatherdir = re.compile("weather", re.IGNORECASE)
        foxdir = re.compile("fox", re.IGNORECASE)
        amcdir = re.compile("amc", re.IGNORECASE)
        hsndir = re.compile("hsn", re.IGNORECASE)
        vh1dir = re.compile("vh1", re.IGNORECASE)
        baddir = re.compile("bad", re.IGNORECASE)
        gooddir = re.compile("good", re.IGNORECASE)
        diffdir = re.compile("diff", re.IGNORECASE)
        deldir = re.compile("delete", re.IGNORECASE)
        wrongdir = re.compile("wrong", re.IGNORECASE)
        no_channeldir = re.compile("no channel", re.IGNORECASE)
        for root, dirs, files in os.walk(thumbs_path):
            for i, nme in enumerate(dirs):
                if (cnndir.search(nme) is not None or
                        abcdir.search(nme) is not None or
                        baddir.search(nme) is not None or
                        gooddir.search(nme) is not None or
                        diffdir.search(nme) is not None or
                        deldir.search(nme) is not None or
                        wrongdir.search(nme) is not None or
                        no_channeldir.search(nme) is not None or
                        histdir.search(nme) is not None or
                        cbsdir.search(nme) is not None or
                        tntdir.search(nme) is not None or
                        espndir.search(nme) is not None or
                        cnbcdir.search(nme) is not None or
                        msnbcdir.search(nme) is not None or
                        weatherdir.search(nme) is not None or
                        foxdir.search(nme) is not None or
                        amcdir.search(nme) is not None or
                        hsndir.search(nme) is not None or
                        vh1dir.search(nme) is not None or
                        usadir.search(nme) is not None or
                        knbcdir.search(nme) is not None or
                        unidir.search(nme) is not None):
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
            logging.info("Error: No image")
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
            self.button2.setText('Next')
            self.button2.setStyleSheet('font-size: 18pt; font-family: Courier;\
                            color: black; background-color: rgb(230,186,142)')
            self.button3 = QtGui.QPushButton(self)
            self.button3.setText('Previous')
            self.button3.setStyleSheet('font-size: 18pt; font-family: Courier;\
                            color: black; background-color: rgb(230,186,142)')
            self.button4 = QtGui.QPushButton(self)
            self.button4.setText('ABC bot-rig')
            self.button4.setStyleSheet('font-size: 18pt; font-family: Courier;\
                            color: black; background-color: rgb(230,186,142)')
            self.button5 = QtGui.QPushButton(self)
            self.button5.setText('CNN bot-rig')
            self.button5.setStyleSheet('font-size: 18pt; font-family: Courier;\
                            color: black; background-color: rgb(230,186,142)')
            self.button6 = QtGui.QPushButton(self)
            self.button6.setText('UNI bot-rig')
            self.button6.setStyleSheet('font-size: 18pt; font-family: Courier;\
                            color: black; background-color: rgb(230,186,142)')
            self.button34 = QtGui.QPushButton(self)
            self.button34.setText('UNI bot-lef')
            self.button34.setStyleSheet('font-size: 18pt;\
                            font-family: Courier;\
                            color: black; background-color: rgb(230,186,142)')
            self.button7 = QtGui.QPushButton(self)
            self.button7.setText('CBS bot-rig')
            self.button7.setStyleSheet('font-size: 18pt; font-family: Courier;\
                            color: black; background-color: rgb(230,186,142)')
            self.button35 = QtGui.QPushButton(self)
            self.button35.setText('CBS bot-lef')
            self.button35.setStyleSheet('font-size: 18pt;\
                            font-family: Courier;\
                            color: black; background-color: rgb(230,186,142)')
            self.button9 = QtGui.QPushButton(self)
            self.button9.setText('Zoom in')
            self.button9.setStyleSheet('font-size: 18pt; font-family: Courier;\
                            color: black; background-color: rgb(230,186,142)')

            self.button10 = QtGui.QPushButton(self)
            self.button10.setText('Zoom out')
            self.button10.setStyleSheet('font-size: 18pt; font-family:\
                                        Courier; color: black;\
                                        background-color: rgb(230,186,142)')
            self.button11 = QtGui.QPushButton(self)
            self.button11.setText('Zoom fix')
            self.button11.setStyleSheet('font-size: 18pt; font-family:\
                                        Courier; color: black;\
                                        background-color: rgb(230,186,142)')
            self.button12 = QtGui.QPushButton(self)
            self.button12.setText('KNBC bot-lef')
            self.button12.setStyleSheet('font-size: 18pt; font-family:\
                                        Courier; color: black;\
                                        background-color: rgb(230,186,142)')
            self.button36 = QtGui.QPushButton(self)
            self.button36.setText('KNBC bot-rig')
            self.button36.setStyleSheet('font-size: 18pt; font-family:\
                                        Courier; color: black;\
                                        background-color: rgb(230,186,142)')
            self.button37 = QtGui.QPushButton(self)
            self.button37.setText('KNBC top-lef')
            self.button37.setStyleSheet('font-size: 18pt; font-family:\
                                        Courier; color: black;\
                                        background-color: rgb(230,186,142)')
            self.button44 = QtGui.QPushButton(self)
            self.button44.setText('KNBC top-rig')
            self.button44.setStyleSheet('font-size: 18pt; font-family:\
                                        Courier; color: black;\
                                        background-color: rgb(230,186,142)')

            self.button13 = QtGui.QPushButton(self)
            self.button13.setText('TNT bot-rig')
            self.button13.setStyleSheet('font-size: 18pt; font-family:\
                                        Courier; color: black;\
                                        background-color: rgb(230,186,142)')
            self.button38 = QtGui.QPushButton(self)
            self.button38.setText('TNT bot-lef')
            self.button38.setStyleSheet('font-size: 18pt; font-family:\
                                        Courier; color: black;\
                                        background-color: rgb(230,186,142)')
            self.button14 = QtGui.QPushButton(self)
            self.button14.setText('ESPN bot-rig')
            self.button14.setStyleSheet('font-size: 18pt; font-family:\
                                        Courier; color: black;\
                                        background-color: rgb(230,186,142)')

            self.button15 = QtGui.QPushButton(self)
            self.button15.setText('Hist bot-rig')
            self.button15.setStyleSheet('font-size: 18pt; font-family:\
                                        Courier; color: black;\
                                        background-color: rgb(230,186,142)')
            self.button16 = QtGui.QPushButton(self)
            self.button16.setText('USA bot-lef')
            self.button16.setStyleSheet('font-size: 18pt; font-family:\
                                        Courier; color: black;\
                                        background-color: rgb(230,186,142)')
            self.button47 = QtGui.QPushButton(self)
            self.button47.setText('USA bot-rig')
            self.button47.setStyleSheet('font-size: 18pt; font-family:\
                                        Courier; color: black;\
                                        background-color: rgb(230,186,142)')
            self.button17 = QtGui.QPushButton(self)
            self.button17.setText('CNBC bot-rig')
            self.button17.setStyleSheet('font-size: 18pt; font-family:\
                                        Courier; color: black;\
                                        background-color: rgb(230,186,142)')
            self.button39 = QtGui.QPushButton(self)
            self.button39.setText('CNBC dif-asp')
            self.button39.setStyleSheet('font-size: 18pt; font-family:\
                                        Courier; color: black;\
                                        background-color: rgb(230,186,142)')

            self.button18 = QtGui.QPushButton(self)
            self.button18.setText('MSNBC top-rig')
            self.button18.setStyleSheet('font-size: 18pt; font-family:\
                                        Courier; color: black;\
                                        background-color: rgb(230,186,142)')
            self.button40 = QtGui.QPushButton(self)
            self.button40.setText('MSNBC dif-asp')
            self.button40.setStyleSheet('font-size: 18pt; font-family:\
                                        Courier; color: black;\
                                        background-color: rgb(230,186,142)')
            self.button45 = QtGui.QPushButton(self)
            self.button45.setText('MSNBC bot-rig')
            self.button45.setStyleSheet('font-size: 18pt; font-family:\
                                        Courier; color: black;\
                                        background-color: rgb(230,186,142)')

            self.button19 = QtGui.QPushButton(self)
            self.button19.setText('Weath top-lef')
            self.button19.setStyleSheet('font-size: 18pt; font-family:\
                                        Courier; color: black;\
                                        background-color: rgb(230,186,142)')
            self.button41 = QtGui.QPushButton(self)
            self.button41.setText('Weath bot-lef')
            self.button41.setStyleSheet('font-size: 18pt; font-family:\
                                        Courier; color: black;\
                                        background-color: rgb(230,186,142)')
            self.button20 = QtGui.QPushButton(self)
            self.button20.setText('Fox bot-lef')
            self.button20.setStyleSheet('font-size: 18pt; font-family:\
                                        Courier; color: black;\
                                        background-color: rgb(230,186,142)')

            self.button21 = QtGui.QPushButton(self)
            self.button21.setText('AMC bot-rig')
            self.button21.setStyleSheet('font-size: 18pt; font-family:\
                                        Courier; color: black;\
                                        background-color: rgb(230,186,142)')

            self.button22 = QtGui.QPushButton(self)
            self.button22.setText('HSN bot-rig')
            self.button22.setStyleSheet('font-size: 18pt; font-family:\
                                        Courier; color: black;\
                                        background-color: rgb(230,186,142)')

            self.button23 = QtGui.QPushButton(self)
            self.button23.setText('VH1 bot-rig')
            self.button23.setStyleSheet('font-size: 18pt; font-family:\
                                        Courier; color: black;\
                                        background-color: rgb(230,186,142)')
            self.button42 = QtGui.QPushButton(self)
            self.button42.setText('VH1 top-lef')
            self.button42.setStyleSheet('font-size: 18pt; font-family:\
                                        Courier; color: black;\
                                        background-color: rgb(230,186,142)')
            self.button43 = QtGui.QPushButton(self)
            self.button43.setText('VH1 dif-asp')
            self.button43.setStyleSheet('font-size: 18pt; font-family:\
                                        Courier; color: black;\
                                        background-color: rgb(230,186,142)')
            self.button46 = QtGui.QPushButton(self)
            self.button46.setText('VH1 bot-lef')
            self.button46.setStyleSheet('font-size: 18pt; font-family:\
                                        Courier; color: black;\
                                        background-color: rgb(230,186,142)')

            self.button25 = QtGui.QPushButton(self)
            self.button25.setText('Bad lodo')
            self.button25.setStyleSheet('font-size: 18pt;\
                            font-family: Courier;\
                            color: black; background-color: rgb(230,186,142)')
            self.button26 = QtGui.QPushButton(self)
            self.button26.setText('Diff logo')
            self.button26.setStyleSheet('font-size: 18pt;\
                            font-family: Courier;\
                            color: black; background-color: rgb(230,186,142)')
            self.button27 = QtGui.QPushButton(self)
            self.button27.setText('No channel')
            self.button27.setStyleSheet('font-size: 18pt;\
                            font-family: Courier;\
                            color: black; background-color: rgb(230,186,142)')
            self.button29 = QtGui.QPushButton(self)
            self.button29.setText('Delete')
            self.button29.setStyleSheet('font-size: 18pt; font-family:\
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
            self.hor_layout3 = QtGui.QHBoxLayout(self)
            self.ver_layout4 = QtGui.QVBoxLayout(self)
            self.ver_layout5 = QtGui.QVBoxLayout(self)
            self.ver_layout6 = QtGui.QVBoxLayout(self)
            self.ver_layout7 = QtGui.QVBoxLayout(self)
            self.ver_layout.addLayout(self.hor_layout3)
            self.hor_layout3.addLayout(self.ver_layout4)
            self.hor_layout3.addLayout(self.ver_layout5)
            self.hor_layout3.addLayout(self.ver_layout6)
            self.hor_layout3.addLayout(self.ver_layout7)

            self.ver_layout4.addWidget(self.button6)
            self.ver_layout4.addWidget(self.button34)
            self.ver_layout4.addWidget(self.button7)
            self.ver_layout4.addWidget(self.button35)
            self.ver_layout4.addWidget(self.button12)
            self.ver_layout4.addWidget(self.button36)
            self.ver_layout4.addWidget(self.button37)
            self.ver_layout4.addWidget(self.button44)
            self.ver_layout4.addWidget(self.button4)
            self.ver_layout4.addWidget(self.button13)
            self.ver_layout4.addWidget(self.button38)
            self.ver_layout4.addWidget(self.button14)

            self.ver_layout5.addWidget(self.button5)
            self.ver_layout5.addWidget(self.button15)
            self.ver_layout5.addWidget(self.button16)
            self.ver_layout5.addWidget(self.button47)
            self.ver_layout5.addWidget(self.button17)
            self.ver_layout5.addWidget(self.button39)
            self.ver_layout5.addWidget(self.button18)
            self.ver_layout5.addWidget(self.button40)
            self.ver_layout5.addWidget(self.button45)
            self.ver_layout5.addWidget(self.button19)
            self.ver_layout5.addWidget(self.button41)
            self.ver_layout5.addWidget(self.button20)
            self.ver_layout5.addWidget(self.button21)

            self.ver_layout6.addWidget(self.button22)
            self.ver_layout6.addWidget(self.button23)
            self.ver_layout6.addWidget(self.button42)
            self.ver_layout6.addWidget(self.button43)
            self.ver_layout6.addWidget(self.button46)
            self.ver_layout6.addWidget(self.button27)
            self.ver_layout6.addWidget(self.button25)
            self.ver_layout6.addWidget(self.button26)
            self.ver_layout6.addWidget(self.button29)
            self.ver_layout6.addWidget(self.button9)
            self.ver_layout6.addWidget(self.button10)
            self.ver_layout6.addWidget(self.button11)

            self.grid_layout.addLayout(self.ver_layout, 0, 2, 1, 1)
            self.ver_spacer = QtGui.QSpacerItem(20, 40)
            self.grid_layout.addItem(self.ver_spacer, 0, 1, 1, 1)

            self.hor_layout.addLayout(self.grid_layout)

            self.button2.clicked.connect(lambda: self.nextImage(self.thumbs))
            self.button3.clicked.connect(lambda:
                                         self.previousImage(self.thumbs))
            self.button4.clicked.connect(lambda: self.abcbr(self.thumbs))
            self.button5.clicked.connect(lambda: self.cnnbr(self.thumbs))
            self.button6.clicked.connect(lambda: self.unibr(self.thumbs))
            self.button34.clicked.connect(lambda: self.unibl(self.thumbs))
            self.button7.clicked.connect(lambda: self.cbsbr(self.thumbs))
            self.button35.clicked.connect(lambda: self.cbsbl(self.thumbs))
            self.button4.clicked.connect(lambda: self.PassLeft())
            self.button5.clicked.connect(lambda: self.PassLeft())
            self.button6.clicked.connect(lambda: self.PassLeft())
            self.button7.clicked.connect(lambda: self.PassLeft())
            self.button9.clicked.connect(lambda: self.scaleImage(1.25))
            self.button10.clicked.connect(lambda: self.scaleImage(0.8))
            self.button11.clicked.connect(lambda: self.stuck())
            self.button12.clicked.connect(lambda: self.knbcbl(self.thumbs))
            self.button36.clicked.connect(lambda: self.knbcbr(self.thumbs))
            self.button37.clicked.connect(lambda: self.knbctl(self.thumbs))
            self.button44.clicked.connect(lambda: self.knbctr(self.thumbs))
            self.button13.clicked.connect(lambda: self.tntbr(self.thumbs))
            self.button38.clicked.connect(lambda: self.tntbl(self.thumbs))
            self.button14.clicked.connect(lambda: self.espnbr(self.thumbs))
            self.button15.clicked.connect(lambda: self.histbr(self.thumbs))
            self.button16.clicked.connect(lambda: self.usabl(self.thumbs))
            self.button47.clicked.connect(lambda: self.usabr(self.thumbs))
            self.button17.clicked.connect(lambda: self.cnbcbr(self.thumbs))
            self.button39.clicked.connect(lambda: self.cnbcda(self.thumbs))
            self.button18.clicked.connect(lambda: self.msnbctr(self.thumbs))
            self.button40.clicked.connect(lambda: self.msnbcda(self.thumbs))
            self.button45.clicked.connect(lambda: self.msnbcbr(self.thumbs))
            self.button19.clicked.connect(lambda: self.weathertl(self.thumbs))
            self.button41.clicked.connect(lambda: self.weatherbl(self.thumbs))
            self.button20.clicked.connect(lambda: self.foxbl(self.thumbs))
            self.button21.clicked.connect(lambda: self.amcbr(self.thumbs))
            self.button22.clicked.connect(lambda: self.hsnbr(self.thumbs))
            self.button23.clicked.connect(lambda: self.vh1br(self.thumbs))
            self.button42.clicked.connect(lambda: self.vh1tl(self.thumbs))
            self.button43.clicked.connect(lambda: self.vh1da(self.thumbs))
            self.button46.clicked.connect(lambda: self.vh1bl(self.thumbs))
            self.button25.clicked.connect(lambda: self.badImage(self.thumbs))
            self.button26.clicked.connect(lambda: self.diffImage(self.thumbs))
            self.button27.clicked.connect(lambda:
                                          self.no_channelImage(self.thumbs))
            self.button29.clicked.connect(lambda:
                                          self.deleteImage(self.thumbs))

            self.button12.clicked.connect(lambda: self.PassLeft())
            self.button14.clicked.connect(lambda: self.PassLeft())
            self.button15.clicked.connect(lambda: self.PassLeft())
            self.button16.clicked.connect(lambda: self.PassLeft())
            self.button17.clicked.connect(lambda: self.PassLeft())
            self.button18.clicked.connect(lambda: self.PassLeft())
            self.button19.clicked.connect(lambda: self.PassLeft())
            self.button20.clicked.connect(lambda: self.PassLeft())
            self.button21.clicked.connect(lambda: self.PassLeft())
            self.button22.clicked.connect(lambda: self.PassLeft())
            self.button23.clicked.connect(lambda: self.PassLeft())
            self.button25.clicked.connect(lambda: self.PassLeft())
            self.button26.clicked.connect(lambda: self.PassLeft())
            self.button27.clicked.connect(lambda: self.PassLeft())
            self.button29.clicked.connect(lambda: self.PassLeft())
            self.button34.clicked.connect(lambda: self.PassLeft())
            self.button35.clicked.connect(lambda: self.PassLeft())
            self.button36.clicked.connect(lambda: self.PassLeft())
            self.button37.clicked.connect(lambda: self.PassLeft())
            self.button38.clicked.connect(lambda: self.PassLeft())
            self.button39.clicked.connect(lambda: self.PassLeft())
            self.button40.clicked.connect(lambda: self.PassLeft())
            self.button41.clicked.connect(lambda: self.PassLeft())
            self.button42.clicked.connect(lambda: self.PassLeft())
            self.button43.clicked.connect(lambda: self.PassLeft())
            self.button44.clicked.connect(lambda: self.PassLeft())
            self.button45.clicked.connect(lambda: self.PassLeft())
            self.button46.clicked.connect(lambda: self.PassLeft())
            self.button47.clicked.connect(lambda: self.PassLeft())

            self.total_images = len(self.thumbs)
            self.passed_images = 0
            self.left_images = self.total_images
            self.nextImage(self.thumbs)

    def stuck(self):
        if self.should_stuck is False:
            self.should_stuck = True
        else:
            self.should_stuck = False

    def scaleImage(self, factor):
        self.scaleFactor *= factor
        self.label.resize(self.scaleFactor * 940, self.scaleFactor * 540)
        self.current_image = self.current_image.scaled(
            self.scaleFactor * 960, self.scaleFactor * 540)
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

    def badImage(self, thumbs):
        self.moveImage(thumbs, "Bad")

    def diffImage(self, thumbs):
        self.moveImage(thumbs, "Diff")

    def no_channelImage(self, thumbs):
        self.moveImage(thumbs, "No channel")

    def deleteImage(self, thumbs):
        self.moveImage(thumbs, "Delete")

    def abcbr(self, thumbs):
        self.moveImage(thumbs, "407_ABC_bottom_right")

    def cnnbr(self, thumbs):
        self.moveImage(thumbs, "432_CNN_bottom_right")

    def unibr(self, thumbs):
        self.moveImage(thumbs, "400_Univision_bottom_right")

    def unibl(self, thumbs):
        self.moveImage(thumbs, "400_Univision_bottom_left")

    def cbsbr(self, thumbs):
        self.moveImage(thumbs, "402_CBS_bottom_right")

    def cbsbl(self, thumbs):
        self.moveImage(thumbs, "402_CBS_bottom_left")

    def knbcbr(self, thumbs):
        self.moveImage(thumbs, "404_KNBC_bottom_right")

    def knbcbl(self, thumbs):
        self.moveImage(thumbs, "404_KNBC_bottom_left")

    def knbctl(self, thumbs):
        self.moveImage(thumbs, "404_KNBC_top_left")

    def knbctr(self, thumbs):
        self.moveImage(thumbs, "404_KNBC_top_right")

    def tntbr(self, thumbs):
        self.moveImage(thumbs, "415_TNT_bottom_right")

    def tntbl(self, thumbs):
        self.moveImage(thumbs, "415_TNT_bottom_left")

    def espnbr(self, thumbs):
        self.moveImage(thumbs, "424_ESPN_bottom_right")

    def histbr(self, thumbs):
        self.moveImage(thumbs, "439_History_bottom_right")

    def usabl(self, thumbs):
        self.moveImage(thumbs, "441_USA_bottom_left")

    def usabr(self, thumbs):
        self.moveImage(thumbs, "441_USA_bottom_right")

    def cnbcbr(self, thumbs):
        self.moveImage(thumbs, "444_CNBC_bottom_right")

    def cnbcda(self, thumbs):
        self.moveImage(thumbs, "444_CNBC_diff_aspect")

    def msnbctr(self, thumbs):
        self.moveImage(thumbs, "446_MSNBC_top_right")

    def msnbcda(self, thumbs):
        self.moveImage(thumbs, "446_MSNBC_diff_aspect")

    def msnbcbr(self, thumbs):
        self.moveImage(thumbs, "446_MSNBC_bottom_right")

    def weathertl(self, thumbs):
        self.moveImage(thumbs, "454_Weather_channel_top_left")

    def weatherbl(self, thumbs):
        self.moveImage(thumbs, "454_Weather_channel_bottom_left")

    def foxbl(self, thumbs):
        self.moveImage(thumbs, "465_Fox_News_bottom_left")

    def amcbr(self, thumbs):
        self.moveImage(thumbs, "479_AMC_bottom_right")

    def hsnbr(self, thumbs):
        self.moveImage(thumbs, "489_HSN_bottom_right")

    def vh1br(self, thumbs):
        self.moveImage(thumbs, "491_VH1_bottom_right")

    def vh1tl(self, thumbs):
        self.moveImage(thumbs, "491_VH1_top_left")

    def vh1bl(self, thumbs):
        self.moveImage(thumbs, "491_VH1_bottom_left")

    def vh1da(self, thumbs):
        self.moveImage(thumbs, "491_VH1_diff_aspect")

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
        logging.info("Cancel move to %s Image" % (dirnme), self.current_index)
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
