# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.8
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_WidgetCreateProject(object):
    def setupUi(self, WidgetCreateProject):
        WidgetCreateProject.setObjectName("WidgetCreateProject")
        WidgetCreateProject.resize(458, 286)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("../../../../../../Downloads/create_new_folder_FILL0_wght400_GRAD0_opsz48.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        WidgetCreateProject.setWindowIcon(icon)
        WidgetCreateProject.setStyleSheet("background:qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgb(0, 46, 43), stop:1 rgb(28, 67, 66));")
        self.DirectoryLabel = QtWidgets.QLabel(WidgetCreateProject)
        self.DirectoryLabel.setGeometry(QtCore.QRect(10, 10, 271, 61))
        self.DirectoryLabel.setStyleSheet("font: 12pt \"Arial\";\n"
"color:white;\n"
"background-color: rgba(255, 255, 255, 0);\n"
"color:white;")
        self.DirectoryLabel.setObjectName("DirectoryLabel")
        self.NameProjectLabel = QtWidgets.QLabel(WidgetCreateProject)
        self.NameProjectLabel.setGeometry(QtCore.QRect(10, 50, 171, 61))
        self.NameProjectLabel.setStyleSheet("font: 12pt \"Arial\";\n"
"color:white;\n"
"background-color: rgba(255, 255, 255, 0);\n"
"color:white;")
        self.NameProjectLabel.setObjectName("NameProjectLabel")
        self.NameScriptLabel = QtWidgets.QLabel(WidgetCreateProject)
        self.NameScriptLabel.setGeometry(QtCore.QRect(10, 110, 251, 51))
        self.NameScriptLabel.setStyleSheet("font: 12pt \"Arial\";\n"
"color:white;\n"
"background-color: rgba(255, 255, 255, 0);\n"
"color:white;")
        self.NameScriptLabel.setObjectName("NameScriptLabel")
        self.DirectoryButton = QtWidgets.QPushButton(WidgetCreateProject)
        self.DirectoryButton.setGeometry(QtCore.QRect(290, 28, 93, 28))
        self.DirectoryButton.setStyleSheet("QPushButton {\n"
"    font: 10pt \"Arial\";\n"
"    background-color: rgba(255, 255, 255, 0);\n"
"    border-radius: 5px;\n"
"    color:white;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: rgba(255, 255, 255, 50);\n"
"    border-radius: 5px;\n"
"}")
        self.DirectoryButton.setObjectName("DirectoryButton")
        self.NameProjectEdit = QtWidgets.QLineEdit(WidgetCreateProject)
        self.NameProjectEdit.setGeometry(QtCore.QRect(180, 65, 121, 31))
        self.NameProjectEdit.setStyleSheet("font: 12pt \"Arial\";\n"
"background-color: rgba(255, 255, 255, 20);\n"
"border-radius: 5px;\n"
"color:white;\n"
"")
        self.NameProjectEdit.setAlignment(QtCore.Qt.AlignCenter)
        self.NameProjectEdit.setObjectName("NameProjectEdit")
        self.NameMainScript = QtWidgets.QLineEdit(WidgetCreateProject)
        self.NameMainScript.setGeometry(QtCore.QRect(265, 120, 121, 31))
        self.NameMainScript.setStyleSheet("font: 12pt \"Arial\";\n"
"background-color: rgba(255, 255, 255, 20);\n"
"border-radius: 5px;\n"
"color:white;\n"
"")
        self.NameMainScript.setAlignment(QtCore.Qt.AlignCenter)
        self.NameMainScript.setObjectName("NameMainScript")
        self.CreateButton = QtWidgets.QPushButton(WidgetCreateProject)
        self.CreateButton.setGeometry(QtCore.QRect(300, 240, 141, 31))
        self.CreateButton.setStyleSheet("QPushButton {\n"
"    font: 10pt \"Arial\";\n"
"    background-color: rgba(255, 255, 255, 0);\n"
"    border-radius: 5px;\n"
"    color:white;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: rgba(255, 255, 255, 50);\n"
"    border-radius: 5px;\n"
"}")
        self.CreateButton.setObjectName("CreateButton")
        self.ErrorNameProject = QtWidgets.QLabel(WidgetCreateProject)
        self.ErrorNameProject.setGeometry(QtCore.QRect(10, 100, 251, 16))
        self.ErrorNameProject.setStyleSheet("font: 10pt \"Arial\";\n"
"color:white;\n"
"background-color: rgba(255, 255, 255, 0);\n"
"color:red;")
        self.ErrorNameProject.setObjectName("ErrorNameProject")
        self.ErrorMainScript = QtWidgets.QLabel(WidgetCreateProject)
        self.ErrorMainScript.setEnabled(True)
        self.ErrorMainScript.setGeometry(QtCore.QRect(10, 160, 321, 16))
        self.ErrorMainScript.setAutoFillBackground(False)
        self.ErrorMainScript.setStyleSheet("font: 10pt \"Arial\";\n"
"color:white;\n"
"background-color: rgba(255, 255, 255, 0);\n"
"color:red;")
        self.ErrorMainScript.setObjectName("ErrorMainScript")
        self.EnvCheckBox = QtWidgets.QCheckBox(WidgetCreateProject)
        self.EnvCheckBox.setGeometry(QtCore.QRect(10, 200, 281, 20))
        self.EnvCheckBox.setStyleSheet("font: 10pt \"Arial\";\n"
"color:white;\n"
"background-color: rgba(255, 255, 255, 0);\n"
"color:white;")
        self.EnvCheckBox.setObjectName("EnvCheckBox")
        self.DockerCheckBox = QtWidgets.QCheckBox(WidgetCreateProject)
        self.DockerCheckBox.setGeometry(QtCore.QRect(10, 180, 171, 20))
        self.DockerCheckBox.setStyleSheet("font: 10pt \"Arial\";\n"
"color:white;\n"
"background-color: rgba(255, 255, 255, 0);\n"
"color:white;\n"
"")
        self.DockerCheckBox.setObjectName("DockerCheckBox")
        self.ProgressBar = QtWidgets.QProgressBar(WidgetCreateProject)
        self.ProgressBar.setGeometry(QtCore.QRect(10, 245, 281, 23))
        self.ProgressBar.setStyleSheet("QProgressBar {\n"
"    border: 1px solid grey;\n"
"    border-radius: 5px;\n"
"    font: 9pt \"Arial\";\n"
"    background-color: rgba(255, 255, 255, 0);\n"
"    color:white;\n"
"    text-align:center;\n"
"}\n"
"\n"
"\n"
"QProgressBar::chunk {\n"
"    background-color: #02b2ab;\n"
"    border-radius: 5px;\n"
"}\n"
"\n"
"")
        self.ProgressBar.setProperty("value", 24)
        self.ProgressBar.setObjectName("ProgressBar")

        self.retranslateUi(WidgetCreateProject)
        QtCore.QMetaObject.connectSlotsByName(WidgetCreateProject)

    def retranslateUi(self, WidgetCreateProject):
        _translate = QtCore.QCoreApplication.translate
        WidgetCreateProject.setWindowTitle(_translate("WidgetCreateProject", "Создание проекта"))
        self.DirectoryLabel.setText(_translate("WidgetCreateProject", "Выберите папку для проекта:"))
        self.NameProjectLabel.setText(_translate("WidgetCreateProject", "Название проекта:"))
        self.NameScriptLabel.setText(_translate("WidgetCreateProject", "Название главного скрипта:"))
        self.DirectoryButton.setText(_translate("WidgetCreateProject", "Открыть"))
        self.CreateButton.setText(_translate("WidgetCreateProject", "Создать "))
        self.ErrorNameProject.setText(_translate("WidgetCreateProject", "Вы не ввели название проекта."))
        self.ErrorMainScript.setText(_translate("WidgetCreateProject", "Вы не ввели название главного скрипта"))
        self.EnvCheckBox.setText(_translate("WidgetCreateProject", "Создать виртуальное окружение"))
        self.DockerCheckBox.setText(_translate("WidgetCreateProject", "Создать Dockerfile"))
