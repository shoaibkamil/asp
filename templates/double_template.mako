
PyObject* double_in_c(PyObject* arr)
{
		PyObject* new_arr = PySequence_List(arr);
		Py_INCREF(new_arr);
		
		Py_ssize_t index = 0;
		for (int i=0; i<${num_items}; i++)
		{	
			PyObject* item = PySequence_GetItem(arr, index);
			PyObject* two = PyFloat_FromDouble(2.0);
			PyObject* newnum = PyNumber_Multiply(item, two);
			Py_INCREF(newnum);
			PySequence_SetItem(new_arr, index, newnum);
			index++;

		}
		return new_arr;
			

}