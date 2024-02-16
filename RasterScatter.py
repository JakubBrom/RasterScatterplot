# -*- coding: utf-8 -*-
"""
/***************************************************************************
 RasterScatterPlot - A QGIS plugin
 Create scatterplot for two rasters
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
							  -------------------
		begin                : 2019-03-29
		git sha              : $Format:%H$
		copyright            : (C) 2019 by Jakub Brom
		email                : jbrom@zf.jcu.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
from PyQt5.QtCore import QSettings, QTranslator, qVersion, QCoreApplication, Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QAction, QVBoxLayout, QDockWidget

from qgis.core import QgsMapLayerProxyModel

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

import numpy as np
import os.path

from scipy.stats import gaussian_kde

from .RasterScatter_plot import RasterScatterPlotParams

# Initialize Qt resources from file resources.py
from .resources import *

# Import the code for the DockWidget
from .RasterScatter_dockwidget import RasterScatterPlotDockWidget


class RasterScatterPlot(QDockWidget):
	"""QGIS Plugin Implementation."""

	def __init__(self, iface, parent=None):
		super(RasterScatterPlot, self).__init__(parent)
		"""Constructor.

		:param iface: An interface instance that will be passed to this class
			which provides the hook by which you can manipulate the QGIS
			application at run time.
		:type iface: QgsInterface
		"""
		# Save reference to the QGIS interface
		self.iface = iface

		# initialize plugin directory
		self.plugin_dir = os.path.dirname(__file__)

		# initialize locale
		locale = QSettings().value('locale/userLocale')[0:2]
		locale_path = os.path.join(
			self.plugin_dir,
			'i18n',
			'RasterScatterPlot_{}.qm'.format(locale))

		if os.path.exists(locale_path):
			self.translator = QTranslator()
			self.translator.load(locale_path)

			if qVersion() > '4.3.3':
				QCoreApplication.installTranslator(self.translator)

		# Declare instance attributes
		self.actions = []
		self.menu = self.tr(u'&Raster Scatterplot')
		# TODO: We are going to let the user set this up in a future iteration
		self.toolbar = self.iface.addToolBar(u'RasterScatterPlot')
		self.toolbar.setObjectName(u'RasterScatterPlot')

		self.pluginIsActive = False
		self.dockwidget = None

	# noinspection PyMethodMayBeStatic
	def tr(self, message):
		"""Get the translation for a string using Qt translation API.

		We implement this ourselves since we do not inherit QObject.

		:param message: String for translation.
		:type message: str, QString

		:returns: Translated version of message.
		:rtype: QString
		"""
		# noinspection PyTypeChecker,PyArgumentList,PyCallByClass
		return QCoreApplication.translate('RasterScatterPlot', message)

	def add_action(
		self,
		icon_path,
		text,
		callback,
		enabled_flag=True,
		add_to_menu=True,
		add_to_toolbar=True,
		status_tip=None,
		whats_this=None,
		parent=None):
		"""Add a toolbar icon to the toolbar.

		:param icon_path: Path to the icon for this action. Can be a resource
			path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
		:type icon_path: str

		:param text: Text that should be shown in menu items for this action.
		:type text: str

		:param callback: Function to be called when the action is triggered.
		:type callback: function

		:param enabled_flag: A flag indicating if the action should be enabled
			by default. Defaults to True.
		:type enabled_flag: bool

		:param add_to_menu: Flag indicating whether the action should also
			be added to the menu. Defaults to True.
		:type add_to_menu: bool

		:param add_to_toolbar: Flag indicating whether the action should also
			be added to the toolbar. Defaults to True.
		:type add_to_toolbar: bool

		:param status_tip: Optional text to show in a popup when mouse pointer
			hovers over the action.
		:type status_tip: str

		:param parent: Parent widget for the new action. Defaults None.
		:type parent: QWidget

		:param whats_this: Optional text to show in the status bar when the
			mouse pointer hovers over the action.

		:returns: The action that was created. Note that the action is also
			added to self.actions list.
		:rtype: QAction
		"""

		icon = QIcon(icon_path)
		action = QAction(icon, text, parent)
		action.triggered.connect(callback)
		action.setEnabled(enabled_flag)

		if status_tip is not None:
			action.setStatusTip(status_tip)

		if whats_this is not None:
			action.setWhatsThis(whats_this)

		if add_to_toolbar:
			self.toolbar.addAction(action)

		if add_to_menu:
			self.iface.addPluginToRasterMenu(
				self.menu,
				action)

		self.actions.append(action)

		return action

	def initGui(self):
		"""Create the menu entries and toolbar icons inside the QGIS GUI."""

		icon_path = ':/plugins/RasterScatter/icon.png'
		self.add_action(
			icon_path,
			text=self.tr(u'Raster Scatterplot'),
			callback=self.run,
			parent=self.iface.mainWindow())

	def setCboxEmpty(self, comboBox):
		"""Setting of empty value (text) in comboBoxes"""

		comboBox.setAdditionalItems([""])
		ind = comboBox.count() - 1
		comboBox.setCurrentIndex(ind)

	def onClosePlugin(self):
		"""Cleanup necessary items here when plugin dockwidget is closed"""

		#print "** CLOSING RasterScatterPlot"

		# disconnects
		self.dockwidget.closingPlugin.disconnect(self.onClosePlugin)

		# remove this statement if dockwidget is to remain
		# for reuse if plugin is reopened
		# Commented next statement since it causes QGIS crashe
		# when closing the docked window:
		self.dockwidget.cb_rast1.clear()
		self.dockwidget.cb_rast2.clear()
		self.dockwidget.cb_mask.clear()
		self.figure.clear()
		self.dockwidget = None

		self.pluginIsActive = False

	def unload(self):
		"""Removes the plugin menu item and icon from QGIS GUI."""

		#print "** UNLOAD RasterScatterPlot"

		for action in self.actions:
			self.iface.removePluginRasterMenu(
				self.tr(u'&RasterScatter'),
				action)
			self.iface.removeToolBarIcon(action)
		# remove the toolbar
		del self.toolbar

	def plot(self):
		"""Plot results of regression analysis and scatterplot"""

		# Read RasterScatterPlotParams() class
		rp = RasterScatterPlotParams()

		# Read raster layer paths:
		## Read raster 1
		try:
			rast1_index = self.dockwidget.cb_rast1.currentIndex()
			path_x = self.dockwidget.cb_rast1.layer(rast1_index).source()
		except Exception:
			path_x = self.dockwidget.cb_rast1.currentText()

		# Read raster 2
		try:
			rast2_index = self.dockwidget.cb_rast2.currentIndex()
			path_y = self.dockwidget.cb_rast2.layer(rast2_index).source()
		except Exception:
			path_y = self.dockwidget.cb_rast2.currentText()

		# Read mask
		try:
			mask_index = self.dockwidget.cb_mask.currentIndex()
			path_mask = self.dockwidget.cb_mask.layer(mask_index).source()
		except Exception:
			path_mask = self.dockwidget.cb_mask.currentText()
		if path_mask == "":
			path_mask = None

		## Create arrays with data
		x_raw = rp.readRaster(path_x, path_mask)		# raster X
		y_raw = rp.readRaster(path_y, path_mask)		# raster Y

		# Values selection - because the drawing of huge number of values in the plot
		x_len = self.dockwidget.sb_select.value()

		if x_len < len(x_raw) and x_len is not None and x_len != 0:
			select_rows = np.random.choice(x_raw.shape[0], size=x_len, replace=False)
			x = x_raw[select_rows]
			y = y_raw[select_rows]
		else:
			x = x_raw
			y = y_raw

		x = np.nan_to_num(x)
		y = np.nan_to_num(y)

		# Define method
		method = self.dockwidget.cb_method.currentIndex()
		if method == None:
			method = 0

		# Plot parameters
		t, z_line, str_r2, equation = rp.regressLineParam(x_raw, y_raw, method)

		# Clear last figure
		self.figure.clear()
		
		# set size of the plot - margins
		self.figure.subplots_adjust(left=0.2, bottom=0.15, right=0.95, top=0.95, wspace=0, hspace=0)

		# create an axis
		ax = self.figure.add_subplot(111)
		ax.set_aspect("auto", "box")

		# Axes names
		x_name = self.dockwidget.cb_rast1.currentText()
		y_name = self.dockwidget.cb_rast2.currentText()

		# Axes labels
		ax.set_xlabel(x_name)
		ax.set_ylabel(y_name)

		# plot data
		try:
			# Calculate the point density
			xy = np.vstack([x, y])
			z = gaussian_kde(xy)(xy)

			# plot the points
			ax.scatter(x, y, c=z, s=5)

		except Exception:
			ax.scatter(x, y, s=5)

		# plot regression line and results
		ax.plot(t, z_line, label = equation + "; " + str_r2, c = "black", linewidth = 2)

		# Legend position
		ax.legend(loc='best')
		
		# Draw plot
		self.canvas.draw()

	def run(self):
		"""Run method that loads and starts the plugin"""

		if not self.pluginIsActive:
			self.pluginIsActive = True
			
			# dockwidget may not exist if:
			#    first run of plugin
			#    removed on close (see self.onClosePlugin method)
			if self.dockwidget == None:
				## Create the dockwidget (after translation) and keep reference
				self.dockwidget = RasterScatterPlotDockWidget()

			# If mask is not used cbox is empty
			self.setCboxEmpty(self.dockwidget.cb_mask)
			self.dockwidget.checkBox.toggled.connect(lambda: self.setCboxEmpty(self.dockwidget.cb_mask))
			self.dockwidget.checkBox_sel.toggled.connect(lambda: self.dockwidget.sb_select.setValue(0))

			# Read items from iface
			# Read list of names and layers paths from QGIS legend
			self.dockwidget.cb_rast1.setFilters(QgsMapLayerProxyModel.RasterLayer)
			self.dockwidget.cb_rast2.setFilters(QgsMapLayerProxyModel.RasterLayer)
			self.dockwidget.cb_mask.setFilters(QgsMapLayerProxyModel.RasterLayer)


			# Create plot
			self.figure = plt.figure(facecolor = "white")									# create plot
			self.canvas = FigureCanvas(self.figure)						# set in to canvas

			# Buttonbox for creating figure
			self.dockwidget.buttonBox.clicked.connect(self.plot)
			
			# Add plot in to UI widget
			lay = QVBoxLayout(self.dockwidget.widget)  
			lay.setContentsMargins(0, 0, 0, 0)      
			lay.addWidget(self.canvas)
			self.setLayout(lay)

			# connect to provide cleanup on closing of dockwidget
			self.dockwidget.closingPlugin.connect(self.onClosePlugin)

			# show the dockwidget
			# TODO: fix to allow choice of dock location
			self.iface.addDockWidget(Qt.LeftDockWidgetArea, self.dockwidget)
			self.dockwidget.show()
