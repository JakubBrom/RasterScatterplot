# -*- coding: utf-8 -*-


#-----------------------------------------------------------------------
# RasterScatter_plot
#
# Author: Jakub Brom, University of South Bohemia in Ceske Budejovice,
#		  Faculty of Agriculture 
# 		Date: 2019-04-06
#		begin                : 2019-03-29
#		copyright            : (C) 2019 by Jakub Brom
#		email                : jbrom@zf.jcu.cz
#
#-----------------------------------------------------------------------
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#-----------------------------------------------------------------------


from osgeo import gdal
import sys, os
import numpy as np


class RasterScatterPlotParams:
	def __init__(self):
		pass
		
	def readRaster(self, rast, rast_mask=None):
		"""
		Function readGeo reads raster and mask files and makes 1d Numpy
		array with the data restricted by the mask.

		Inputs:
		:param rast: Path to raster file in GDAL readable format
		:type rast: str
		:param rast_mask: Path to raster mask file in GDAL readable
						  format.
						  Data are typicaly 0 (nodata) and 1 (data).
		:type rast_mask: str
		
		Returns:
		:returns: 1D Numpy array with data from raster or data from
				  raster restricted by mask
		:rtype: Numpy array, float32
		"""
		
		# raster processing
		ds = gdal.Open(rast)
		try:
			array_in = gdal.Dataset.ReadAsArray(ds).astype(np.float32)
		except:
			raise IOError('Error reading raster data. File might be too big.')
		ds = None
		
		# mask processing (0 = nodata, 1 = data)TODO: vyresit problem s nactenim dat, ktera nemaji binarni charakter --> prevod?
		if rast_mask != None:
			dsm = gdal.Open(rast_mask)
			try:
				array_mask = gdal.Dataset.ReadAsArray(dsm).astype(np.int8)
			except:
				raise IOError('Error reading raster data. File might be too big.')
			dsm = None

			rast_1d = np.ravel(array_in)
			rast_1d_mask = np.ravel(array_mask)                    # flattening of the data
			mask_bool = np.ma.make_mask(rast_1d_mask)              # transformation of the mask in to the boolean
			rast_1d = rast_1d[mask_bool]
			
			#rast_1d = array_in[mask_bool] = True		               # exclusion of the "nodata" from the raster data
		else:
			rast_1d = np.ravel(array_in)                              # flattening of the data
	
		return rast_1d


	def linRegresCoeff(self, data_x, data_y):
		"""Calculation of linear regression coefficients
		Inputs:
		:param data_x: Data for X axis
		:type data_x: list, 1D Numpy array
		:param data_y: Data for Y axis
		:type data_y: list, 1D Numpy array
		
		Returns:
		:returns slope: Slopes coeff. of linear regression.
		:rtype list_slope: float
		:returns intercept: Intercept coeff. of linear regression.
		:rtype list_intercept: float
		:returns r: Pearsson correlation coeff. of linear regression.
		:rtype r: float
		"""
		
		denom = data_x.dot(data_x) - data_x.mean() * data_x.sum()
		slope = (data_x.dot(data_y) - data_y.mean() * data_x.sum()) / denom
		intercept = (data_y.mean() * data_x.dot(data_x) - data_x.mean() * data_x.dot(data_y)) / denom
		r = np.corrcoef([data_x, data_y])[0,1]
	
		return slope, intercept, r

	
	def regressAll(self, data_x, data_y, method = 0):
		"""
		Regression analysis between two rasters. Analysis is calculated
		for all the linear,	logarithmic, exponential and power models.
		Outputs are lists with results of the models.

		Inputs:
		:param data_x: Data for X axis
		:type data_x: list, 1D Numpy array
		:param data_y: Data for Y axis
		:type data_y: list, 1D Numpy array
		:param method: Method of calculation: 0 - automatic selection
					   of a model with best fit; 1 - linear; 2 - natural
					   logarithm; 3 - exponential; 4 - power
		:type method: int
		
		Returns:
		:returns list_slope: List of slopes for all regression methods.
		:rtype list_slope: list
		:returns list_intercept: List of intercepts for all regression
								 methods.
		:rtype list_intercept: list
		:returns list_r: List of r values for all regression methods.
		:rtype list_r: list
		"""
	
		ignore_zero = np.seterr(all = "ignore")
		
		list_x = [data_x, np.log(data_x), data_x, np.log(data_x)]
		list_y = [data_y, data_y, np.log(data_y), np.log(data_y)]
		
		# Calculation of constants for regression models: slope, intercept, r and p
		list_r = [int() for i in range(4)]                               	# list of r values for the methods
		list_slope = [int() for i in range(4)]								# list of slopes
		list_intercept = [int() for i in range(4)]							# list of intercepts
		
		# Models - calculating regressions
		if method == 0:
			for i in range(0, len(list_x)):
				slope_lin, inter_lin, r_lin = self.linRegresCoeff(list_x[i],list_y[i])
				list_slope[i] = slope_lin
				list_r[i] = r_lin
				if i < 2:
					list_intercept[i] = inter_lin
				else:
					list_intercept[i] = np.exp(inter_lin)
		else:
			slope_lin, inter_lin, r_lin = self.linRegresCoeff(list_x[method-1],list_y[method-1])
			list_slope[method-1] = slope_lin
			list_r[method-1] = r_lin
			if method-1 < 2:
				list_intercept[method-1] = inter_lin
			else:
				list_intercept[method-1] = np.exp(inter_lin)
			
		return list_slope, list_intercept, list_r

		
	def choseModelParam(self, method, list_slope, list_intercept, list_r):
		"""
		Setting regresion model according to max r-value for best-fit
		method.

		Inputs:
		:param method: Method of calculation: 0 - automatic selection
					   of a model with best fit; 1 - linear; 2 - natural
					   logarithm; 3 - exponential; 4 - power
		:type method: int
		:param list_slope: List of slopes for all regression methods.
		:type list_slope: list
		:param list_intercept: List of intercepts for all regression
								 methods.
		:type list_intercept: list
		:param list_r: List of r values for all regression methods.
		:type list_r: list

		Returns:
		:returns model: Regression model: 0 - linear; 1 - natural
						logarithm; 2 - exponential; 3 - power
		:rtype model: int
		:returns slope: Slopes coeff. of linear regression.
		:rtype list_slope: float
		:returns intercept: Intercept coeff. of linear regression.
		:rtype list_intercept: float
		:returns r2: Regression coefficient.
		:rtype r2: float
		"""

		list_r2 = list(np.array(list_r) ** 2)
		
		if method == 0:
			model = list_r2.index(max(list_r2[:]))		# setting of regression model according to max r2 value
		else:
			model = method - 1
			
		slope = list_slope[model]
		intercept = list_intercept[model]
		r = list_r[model]
		r2 = r**2
		
		return model, slope, intercept, r2 
	
	
	def regressLineParam(self, x, y, method = 0):
		"""
		Function manages parameters for drawing scatterplot for two
		rasters and parameters of corresponding regression line.
		The regression equation and determination coefficient
		are written in the legend of the graph.
		
		Inputs:
		:param x: Data for X axis
		:type x: list, 1D Numpy array
		:param y: Data for Y axis
		:type y: list, 1D Numpy array
		:param method: Method of calculation: 0 - automatic selection
					   of a model with best fit; 1 - linear; 2 - natural
					   logarithm; 3 - exponential; 4 - power
		:type method: int

		Returns:
		:returns t: Sequention of scaled data for axis X.
		:rtype t: 1D Numpy array
		:returns z_line: Regression line data corresponding to X axis.
		:rtype z_line: 1D Numpy array
		:returns str_r2: Text "R2 = ?" added in the plot legend.
		:rtype str_r2: str (LaTEX code)
		:returns equation: Text of equation used in plot legend.
		:rtype equation: str (LaTEX code)
		"""

		# calculate parameters for all methods
		list_slope, list_intercept, list_r = self.regressAll(x, y, method)

		# chose coefficients according to model used
		model, slope, intercept, r2 = self.choseModelParam(method, list_slope, list_intercept, list_r)

		# Model name
		#rmethod_name = ["Linear", "Nat. log.", "Exponential", "Power"]
		#model_name = rmethod_name[model]

		# x axis for regression line
		t = np.linspace(min(x), max(x), 300)
		
		if model == 0:
			z_line = intercept + slope * t																# linear fit
			round_int = np.round(intercept, 3)
			round_sl = np.round(slope, 3)
			if round_sl > 0.0 and round_int != 0.0:
				equation = r"$y={%s+x}{%s}$" % (round_int, round_sl)
			elif round_sl == 0.0 and round_int != 0.0:
				equation = r"$y={%s}$" % (round_int)
			elif round_sl != 0.0 and round_int == 0.0:
				equation = r"$y={%sx}$" % (round_sl)
			elif round_sl < 0.0 and round_int != 0.0:
				equation = r"$y={%s%sx}$" % (round_int, round_sl)
			else:
				equation = r"$y={0.0}$"
			
		else:
			if model == 1:
				z_line = intercept + slope * np.log(t)													# nat. log. fit
				round_int = np.round(intercept, 3)
				round_sl = np.round(slope, 3)
				if round_sl > 0.0 and round_int != 0.0:
					equation = r"$y={%s}+{%s}\ln{x}$" % (round_int, round_sl)
				elif round_sl == 0.0 and round_int != 0.0:
					equation = r"$y={%s}$" % (round_int)
				elif round_sl != 0.0 and round_int == 0.0:
					equation = r"$y={%s}\ln{x}$" % (round_sl)
				elif round_sl < 0.0 and round_int != 0.0:
					equation = r"$y={%s}{%s}\ln{x}$" % (round_int, round_sl)
				else:
					equation = r"$y={0.0}$"
					
			else:
				if model == 2:
					z_line = intercept * np.exp(slope * (t))												# exp. fit
					round_int = np.round(intercept, 3)
					round_sl = np.round(slope, 5)
					equation = r"$y={%s}\mathrm{e}^{{%s}{x}}$" % (round_int, round_sl)
				else:
					z_line = intercept * t ** slope														# pow. fit
					round_int = np.round(intercept, 3)
					round_sl = np.round(slope, 5)
					equation = r"$y={%s}{x}^{%s}$" % (round_int, round_sl)
		
		if r2 == 0.0:
			round_r2 = 0.0
		else:
			round_r2 = np.round(r2, 3)
		
		str_r2 = r"$r^2=%s$" % str(round_r2)
		
		return t, z_line, str_r2, equation
