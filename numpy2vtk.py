
#! /usr/bin/env python

# Note: as the pypi version seems to be broken, I copied the github version


# ***********************************************************************************
# * Copyright 2010 - 2016 Paulo A. Herrera. All rights reserved.                    *
# *                                                                                 *
# * Redistribution and use in source and binary forms, with or without              *
# * modification, are permitted provided that the following conditions are met:     *
# *                                                                                 *
# *  1. Redistributions of source code must retain the above copyright notice,      *
# *  this list of conditions and the following disclaimer.                          *
# *                                                                                 *
# *  2. Redistributions in binary form must reproduce the above copyright notice,   *
# *  this list of conditions and the following disclaimer in the documentation      *
# *  and/or other materials provided with the distribution.                         *
# *                                                                                 *
# * THIS SOFTWARE IS PROVIDED BY PAULO A. HERRERA ``AS IS'' AND ANY EXPRESS OR      *
# * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF    *
# * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO      *
# * EVENT SHALL <COPYRIGHT HOLDER> OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,        *
# * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,  *
# * BUT NOT LIMITED TO, PROCUREMEN OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,    *
# * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY           *
# * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING  *
# * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS              *
# * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                    *
# ***********************************************************************************

# **************************************************************
# * Example of how to use the high level imageToVTK function.  *
# **************************************************************
from pyevtk.vtk import (
	VtkFile,
	VtkUnstructuredGrid,
	VtkImageData,
	VtkRectilinearGrid,
	VtkStructuredGrid,
	VtkVertex,
	VtkLine,
	VtkPolyLine,
	VtkPixel,
)
def _addDataToFile(vtkFile, cellData, pointData, fieldData=None):
	# Point data
	if pointData:
		keys = list(pointData.keys())
		vtkFile.openData("Point", scalars=keys[0])
		for key in keys:
			data = pointData[key]
			vtkFile.addData(key, data)
		vtkFile.closeData("Point")

	# Cell data
	if cellData:
		keys = list(cellData.keys())
		vtkFile.openData("Cell", scalars=keys[0])
		for key in keys:
			data = cellData[key]
			vtkFile.addData(key, data)
		vtkFile.closeData("Cell")

	# Field data
	# https://www.visitusers.org/index.php?title=Time_and_Cycle_in_VTK_files#XML_VTK_files
	if fieldData:
		keys = list(fieldData.keys())
		vtkFile.openData("Field")  # no attributes in FieldData
		for key in keys:
			data = fieldData[key]
			vtkFile.addData(key, data)
		vtkFile.closeData("Field")


def _appendDataToFile(vtkFile, cellData, pointData, fieldData=None):
	# Append data to binary section
	if pointData is not None:
		keys = list(pointData.keys())
		for key in keys:
			data = pointData[key]
			vtkFile.appendData(data)

	if cellData is not None:
		keys = list(cellData.keys())
		for key in keys:
			data = cellData[key]
			vtkFile.appendData(data)

	if fieldData is not None:
		keys = list(fieldData.keys())
		for key in keys:
			data = fieldData[key]
			vtkFile.appendData(data)

def imageToVTK(
	path,
	origin=(0.0, 0.0, 0.0),
	spacing=(1.0, 1.0, 1.0),
	cellData=None,
	pointData=None,
	fieldData=None,
	):
	"""
	Export data values as a rectangular image.
	Parameters
	----------
	path : str
		name of the file without extension where data should be saved.
	origin : tuple, optional
		grid origin.
		The default is (0.0, 0.0, 0.0).
	spacing : tuple, optional
		grid spacing.
		The default is (1.0, 1.0, 1.0).
	cellData : dict, optional
		dictionary containing arrays with cell centered data.
		Keys should be the names of the data arrays.
		Arrays must have the same dimensions in all directions and can contain
		scalar data ([n,n,n]) or vector data ([n,n,n],[n,n,n],[n,n,n]).
		The default is None.
	pointData : dict, optional
		dictionary containing arrays with node centered data.
		Keys should be the names of the data arrays.
		Arrays must have same dimension in each direction and
		they should be equal to the dimensions of the cell data plus one and
		can contain scalar data ([n+1,n+1,n+1]) or
		+1,n+1,n+1],[n+1,n+1,n+1],[n+1,n+1,n+1]).
		The default is None.
	fieldData : dict, optional
		dictionary with variables associated with the field.
		Keys should be the names of the variable stored in each array.
	Returns
	-------
	str
		Full path to saved file.
	Notes
	-----
	At least, cellData or pointData must be present
	to infer the dimensions of the image.
	"""
	assert cellData is not None or pointData is not None

	# Extract dimensions
	start = (0, 0, 0)
	end = None
	if cellData is not None:
		keys = list(cellData.keys())
		data = cellData[keys[0]]
		if hasattr(data, "shape"):
			end = data.shape
		elif data[0].ndim == 3 and data[1].ndim == 3 and data[2].ndim == 3:
			end = data[0].shape
	elif pointData is not None:
		keys = list(pointData.keys())
		data = pointData[keys[0]]
		if hasattr(data, "shape"):
			end = data.shape
		elif data[0].ndim == 3 and data[1].ndim == 3 and data[2].ndim == 3:
			end = data[0].shape
		end = (end[0] - 1, end[1] - 1, end[2] - 1)

	# Write data to file
	w = VtkFile(path, VtkImageData)
	w.openGrid(start=start, end=end, origin=origin, spacing=spacing)
	w.openPiece(start=start, end=end)
	_addDataToFile(w, cellData, pointData, fieldData)
	w.closePiece()
	w.closeGrid()
	_appendDataToFile(w, cellData, pointData, fieldData)
	w.save()
	return w.getFileName()
 
