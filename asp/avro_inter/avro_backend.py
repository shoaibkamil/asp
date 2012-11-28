
"""
TO NOTE:
1) args must be entered into original python call in the same order they are used in the mainfunc
    given
2) must have matching number of args in args.avro as are required to call mainfunc, in proper order
3) mainfunc name needs to be the same as the name for the function added to the scala module
"""

import py_avro_inter as py_avro

def generate_scala_object(mainfunc, filename=None, rendered=None):   
    
    class_name = mainfunc + "_outer"
    if not rendered and filename:
        f = open(filename)
        rendered = f.read()
        f.close()
    
    output = """
import javro.JAvroInter
import org.apache.avro.Schema
import javro.scala_arr

class %s{
    %s
}
    """ %(class_name + "_data", rendered)
    
    output+= """

object %s{ 
    """%(class_name)
    output += generate_scala_main(rendered, mainfunc)    
    output += """
}
"""
    #print 'output is ', output
    return output


#multiple outputs????
def generate_scala_main(rendered, mainfunc):
    
    main = """
    def main(args: Array[String]){  
        var s = new JAvroInter("results.avro", "args.avro") 
        var results = new Array[Object](1)
        %s
        s.writeAvroFile(results)   
                
    }
    """ %(generate_func_call(rendered,mainfunc))
    return main

def generate_func_call(rendered, mainfunc):
    size = get_arg_amount(rendered, mainfunc)
    arg_types = get_arg_type(rendered, mainfunc)
    call = ""
    args = ""
    for i in range(size):
        call += """var arg%s = s.returnStored[%s](%s)
        """ %(i, arg_types[i], i) 
        args += "arg%s" %i
        if not i== (size-1):
            args+=', '    
    call += "results(0) = %s(%s).asInstanceOf[Object]" %('(new ' +mainfunc +'_outer_data()).' + mainfunc, args)
    return call

def get_arg_type(rendered, mainfunc):
    size = get_arg_amount(rendered, mainfunc)
    start_index = rendered.find(mainfunc)
    args_found = 0
    colon_indices=[]
    while (args_found < size):
        if colon_indices:
            colon_indices.append(rendered.index(':', colon_indices[-1]+1))
        else:
            colon_indices.append(rendered.index(':', start_index))
        args_found += 1
    types = parse_func(rendered, colon_indices, mainfunc)  
    return types  
    #return "Int"


def parse_func(rendered, colon_indices, mainfunc):
    comma_indices = []
    count = 0
    while len(comma_indices) < (len(colon_indices) -1):
        comma_indices.append(rendered.index(',', colon_indices[count]))
        count += 1 
    types = []
    count = 0
    while (len(types) < len(colon_indices)):
        if (len(types)==(len(colon_indices)-1)):
            types.append(rendered[colon_indices[count]+1:closing_paren_loc(rendered, rendered.find(mainfunc))])
        else:
            types.append(rendered[colon_indices[count]+1 : comma_indices[count]])
        count += 1
    #arg types are now between : and ,'s
    return types

#calculates arg amount in function simply by counting the commas...could be more rigorous, but
#can't think of a situation in which it won't work
def get_arg_amount(rendered, mainfunc):
    start = opening_paren_loc(rendered,rendered.find(mainfunc))
    end = closing_paren_loc(rendered,rendered.find(mainfunc))    
    index = start
    comma_count = 0
    while index < end:
        char = rendered[index]
        if char == ',':
            comma_count +=1
        index+=1
    return comma_count+1

def opening_paren_loc(str, start_index):
    index = start_index
    while index < len(str):
        char = str[index]
        if char == '(':
            return index
        else: index +=1
    return index

#returns the index of the closin paren
#first_paren is the index ideally of the opening paren, but still works if index is 
# before the first paren (of course assuming there aren't other parens in between

def closing_paren_loc(str, first_paren):
    paren_count = -1
    index = first_paren   
    while index < len(str):
        char = str[index]
        if char == ')' and paren_count == 0:
            return index      
        elif char == '(':
            paren_count+=1        
        elif char == ')':
            paren_count -= 1
        index+=1
    raise "No closing paren found"

if __name__ == '__main__':
    print "beginning"
    print(generate_scala_object('double', 'func1.scala'))
    print"DONE"
    
