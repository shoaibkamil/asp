using namespace boost::python;

void map_in_c(list arr)
{
    for (int i=0; i < ${num_items}; i++)
    {
	double x = extract<double>(arr[i]);
	x = ${expr};
	arr[i] = x;
    }
}
