import sys
from avro import schema, datafile, io
from cStringIO import StringIO


"""
Module to read from and write to .avro files

TO NOTE:
1) lists can only be of one type
2) tuples are converted to lists
"""

stored = []

def getAvroType(pyObject):
    t = type(pyObject)
    if t == dict:
        return '"record"'
    elif t == list or t == tuple:
        if pyObject:
            listType = getAvroType(pyObject[0])
        else:
            #list is empty...
            listType = '"int"'
        entry = """{    "type":"array", "items": %s    }"""%(listType)
        return entry
    elif t == str:
        return '"string"'
    elif t == int:
        return '"int"'
    elif t == long:
        return '"long"'
    elif t == float:
        return '"double"'
    elif t == bool:
        return '"boolean"'
    elif t == type(None):
        return '"null"'
    else:
        raise Exception("Unrecognized type")
    return entry
        

def makeSchema(args):
    schema = """{
    "type": "record",
    "name": "args",
    "namespace": "SCALAMODULE",
    "fields": ["""
    count = 1
    size = """
        { "name": "size"    , "type": "int"    }"""
    if args:
        size += ","            
    schema = schema +size
    for arg in args:
        t = getAvroType(arg)
        entry = """
        {    "name": "arg%s"    , "type": %s    }"""%(count,t)
        if count != len(args):
            entry+= ','
        schema = schema + entry
        count+=1
    close = """
    ]
}"""
    schema = schema + close
    return schema

    
def write_avro_file(args, outsource='args.avro'):
    SCHEMA = schema.parse(makeSchema(args))
    rec_writer = io.DatumWriter(SCHEMA)   
        
    if outsource == sys.stdout:
        df_writer = datafile.DataFileWriter(sys.stdout, rec_writer, 
                                        writers_schema = SCHEMA, codec = 'deflate')
    
    else:
        df_writer = datafile.DataFileWriter(open(outsource,'wb'), rec_writer, 
                                        writers_schema = SCHEMA, codec = 'deflate')
    data = {}
    count = 1
    data['size'] = len(args)
    for arg in args:
        if type(arg) == tuple:
            arg = tupleToList(arg)
        data["arg%s"%(count)] = arg
        count +=1
    df_writer.append(data)
    df_writer.close()

#this function reads the specified avro file and stores the data in the global list stored
def read_avro_file(insource='results.avro'):
    rec_reader = io.DatumReader()
    if insource == sys.stdin:          
        input = sys.stdin.read()
        temp_file = StringIO(input)

        df_reader = datafile.DataFileReader(temp_file, rec_reader)
    else:
        df_reader = datafile.DataFileReader(open(insource), rec_reader)
    del stored[:]
    """
    for record in df_reader:
        size = record['size']
        for i in range(size):
            i = i+1
            arg = record["arg%s"%(i)]
            #print arg
            stored.append(arg)
    """
    return df_reader

def return_stored(index):
    if stored:
        return stored[index]
    else:
        read_avro_file()
        return stored[index]
    
def return_stored():
    if stored:
        return stored
    else:
        read_avro_file()
        return stored
        
def tupleToList(input):
    output = list(input)
    for i in range(len(output)):
        if type(output[i]) == tuple:
            output[i] = list(output[i])
    return output
        

if __name__ == '__main__': 
    args = sys.argv   
    #inputs = [[1.0*i for i in xrange(10000000)]]
    inputs = [1,2,[3,34]]
    import time
    print "about to write"
    start = time.time()
    write_avro_file(inputs)
    end = time.time()
    print "done writing"
    res = read_avro_file('args.avro')
    print 'FROM FILE:' + str(res)
    
