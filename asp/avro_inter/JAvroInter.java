package javro;

import java.util.List;


import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

import org.apache.avro.Schema;
import org.apache.avro.file.DataFileReader;
import org.apache.avro.file.DataFileWriter;
import org.apache.avro.file.DataFileStream;
import org.apache.avro.generic.GenericData;
import org.apache.avro.generic.GenericDatumReader;
import org.apache.avro.generic.GenericDatumWriter;
import org.apache.avro.generic.GenericRecord;
import org.apache.avro.io.DatumReader;
import org.apache.avro.io.DatumWriter;
import org.apache.avro.util.Utf8;
import java.io.*;


/**
 * 
 * TO NOTE: 
 * 	1) Cannot write arrays. Can only write objects that inherit from java.util.List. This is due to 
 * 		the fact that the avro machinery tries to cast all inputs declared as arrays to java.util.List
 * 		...looking for a way around
 * 	2) input args are put in an Object[T],in which arrays are stored as GenericData.Array[T],
 * 		and strings are of class Utf8
 * 
 * 	3) when calling returnStored(index), must use the wrapper classes not primitive types. 
 * 		i.e. write Integer i = xed(index) instead of int i = x.returnStored(index)
 * 		or write Double d = x.returnStored(index) instead of double d = x.returnStored(index)
 *
 */

public class JAvroInter{
	
	String OUTPUT_FILE_NAME;
	String INPUT_FILE_NAME;
	Schema schema;
	public Object[] stored;
	
	public JAvroInter(String outputFile, String inputFile) throws IOException, IllegalAccessException,InstantiationException,ClassNotFoundException{
		OUTPUT_FILE_NAME = outputFile;
		INPUT_FILE_NAME = inputFile;

	}
	
	public String getAvroType(Object item){
		if (item == null){
			return "\"null\"";
		}
		Class c = item.getClass();
		String name = c.getName();
		/**
		if (name == "java.util.Arrays$ArrayList" || name == "scala.collection.JavaConversions$MutableBufferWrapper" 
				|| name == "org.apache.avro.generic.GenericData$Array"){
		**/
		if (item instanceof List){
			return String.format("{ \"type\":\"array\", \"items\": %s}", getAvroType(((List)item).get(0)));
		}		 
		else if (name == "java.lang.Double" || name == "double"){
			return "\"double\"";
		}
		else if (name == "java.lang.Integer" || name == "int"){
			return "\"int\"";
		}
		else if (name == "java.lang.String" || name == "org.apache.avro.util.Utf8"){
			return "\"string\"";
		}
		else if (name == "java.lang.Float" || name == "float"){
			return "\"float\"";
		}
		else if (name == "java.lang.Long" || name == "long"){
			return "\"long\"";
		}
		else if (name == "java.lang.Boolean" || name == "boolean"){
			return "\"boolean\"";
		}
		else {
			System.out.println(name);
			throw new RuntimeException("Unknown Argument Type to Write to Avro File");
		}
	}
	
	
	public String makeSchema(Object[] args){
		String schema = "{\n"
			+"\t\"type\": \"record\",\n"
			+"\t\"name\": \"args\",\n"
			+"\t\"namespace\": \"JAVAMODULE\",\n"
			+"\t\"fields\": [\n";		
		String size = "\t\t{ \"name\": \"size\"	, \"type\": \"int\"	}";
		if (args.length > 0){
			size += ",";
		}
		schema += size;
		String type ="";
		String entry ="";
		int count = 1;
		for (Object arg: args){
			type = this.getAvroType(arg);
			entry = String.format("\n"
				+"\t\t{ \"name\": \"arg%d\"	, \"type\": %s	}", count, type);
			if (count != args.length){
				entry += ",";
			}
			schema += entry;
			count += 1;
		}
		String close = "\n"
			+ "\t]\n}";
		schema += close;
		return schema;
	}
	
	public String makeModelSchema(int length){
		String schema = "{\n"
			+"\t\"type\": \"record\",\n"
			+"\t\"name\": \"args\",\n"
			+"\t\"namespace\": \"JAVAMODULE\",\n"
			+"\t\"fields\": [\n";		
		String size = "\t\t{ \"name\": \"size\"	, \"type\": \"int\"	}";
		size += ",";
		schema += size;
		String type, entry;
		type ="{    \"type\":\"array\", \"items\": \"float\"    }";
		int count = 1;
		
		for (int i=0; i < length; i++){
			entry = String.format("\n"
				+"\t\t{ \"name\": \"arg%d\"	, \"type\": %s	}", count, type);
			if (count != length){
				entry += ",";
			}
			schema += entry;
			count += 1;
		}
		String close = "\n"
			+ "\t]\n}";
		schema += close;
		return schema;
	}
	
	public void writeAvroFile(Object[] args) throws IOException{
		
		String s= this.makeSchema(args);		
		Schema schema = (new Schema.Parser()).parse(s);		
		this.schema = schema;
		
		GenericRecord datum = new GenericData.Record(schema);
		
		datum.put("size", args.length);				
		int count = 1;
		for (Object arg: args){	
			datum.put(String.format("arg%d",count), arg);
			count++;
		}				
		DatumWriter<GenericRecord> writer = new GenericDatumWriter<GenericRecord>(schema);
		DataFileWriter<GenericRecord> dataFileWriter = new DataFileWriter<GenericRecord>(writer);
		if (OUTPUT_FILE_NAME == "System.out"){
			dataFileWriter.create(schema,System.out);
		}
		else {
			File file = new File(OUTPUT_FILE_NAME);
			dataFileWriter.create(schema,file);
		}
		dataFileWriter.append(datum);
		dataFileWriter.close();		
	}	
	
	
	
    public void writeModel(String filename, int num_vecs) throws IOException{

        String s= this.makeModelSchema(1);
        Schema schema = (new Schema.Parser()).parse(s);
        this.schema = schema;


         BufferedReader buffer = new BufferedReader(new FileReader(filename));
         String line = null;
         int count = 1;
         int classes_num = 0;
         int features_num = 0;
         int num = 0;
         float weight =new Float(0.0);
         String[] concat_model;
         while(count < 15)
         {
                line = buffer.readLine();
                if (count == 2){
                        classes_num = Integer.parseInt(line.substring(0, line.indexOf(' ')));
                }
                if (count == 3){
                        features_num = Integer.parseInt(line.substring(0, line.indexOf(' ')));
                }
                 count += 1;
         }

    String elem = "";
    char c;
    int class_count =0;
    int elem_counter = 0;
    int i;

    GenericRecord datum = new GenericData.Record(schema);
    DatumWriter<GenericRecord> writer = new GenericDatumWriter<GenericRecord>(schema);
    DataFileWriter<GenericRecord> dataFileWriter = new DataFileWriter<GenericRecord>(writer);
    File file = new File(OUTPUT_FILE_NAME);

    dataFileWriter.create(schema, file);

    List<Float> vec = new ArrayList<Float>();

    datum = new GenericData.Record(schema);
    datum.put("size", features_num);

    List<Float> pair;
    vec = new ArrayList<Float>();
    while ((i=buffer.read())!= -1){
            c = (char)i;
            if (c != ' '){
                    elem += c;
            }else{
                    if (elem_counter !=0 && elem_counter != 1 && !elem.equals("#")){
                            num = Integer.parseInt(elem.substring(0, elem.indexOf(':')));
                            weight =  java.lang.Float.parseFloat(elem.substring(elem.indexOf(':')+1, elem.length()));
                            pair = new ArrayList<Float>();
                            //((ArrayList)pair).add(num);
                            //((ArrayList)pair).add(weight);
                            ((ArrayList)vec).add(new Float((num-1)%features_num+1));
                            ((ArrayList)vec).add(weight);
                            if ((num) / features_num > class_count){
                                datum.put("arg1", vec);
                                dataFileWriter.append(datum);
                                datum = new GenericData.Record(schema);
                                datum.put("size", features_num);
                                System.out.println("count is:" + class_count);
                                System.out.println("num is:"+ num + "with fn: "+ features_num + "and num/fn is:"+ num/features_num);
                                vec = new ArrayList<Float>();
                     class_count += 1;
                  }
                }
                elem_counter += 1;
                elem = "";
        }
        }
    if ( !((num/features_num) > class_count)){
        datum.put("arg1", vec);
        dataFileWriter.append(datum);
    }

    dataFileWriter.close();

}

                       
                                
	public void readAvroFile() throws IOException, ClassNotFoundException, IllegalAccessException,InstantiationException{
		File file = new File(INPUT_FILE_NAME);
		DatumReader<GenericRecord> reader = new GenericDatumReader<GenericRecord>();

		GenericRecord record;
		if (INPUT_FILE_NAME == "System.in"){
			DataFileStream dfs = new DataFileStream(System.in, reader);
			record = (GenericRecord)dfs.next();
		}
		else{
			DataFileReader<GenericRecord> dataFileReader = new DataFileReader<GenericRecord>(file,reader);
			record = dataFileReader.next();
		}				
		this.store(record);
	}
	
	public DataFileReader<GenericRecord> readModel(String filename)throws IOException, ClassNotFoundException, IllegalAccessException,InstantiationException{
		
		File file = new File(filename);
		DatumReader<GenericRecord> reader = new GenericDatumReader<GenericRecord>();
		GenericRecord record;
		DataFileReader<GenericRecord> dataFileReader = new DataFileReader<GenericRecord>(file,reader);

		return dataFileReader;
		
	}
	/**
	 * this method takes the input data, presumably from args.avro, and stores it in the array stored
	 */	
	public void store(GenericRecord record) throws InstantiationException, IllegalAccessException{
		int size = (java.lang.Integer)record.get("size");
		stored = new Object[size];
		Object item;
		for (int i=0; i < size; i++){
			item = record.get(String.format("arg%d",i+1));
			if (item instanceof org.apache.avro.util.Utf8){
				stored[i] = item.toString();
			}
			else{
				stored[i] = item;
			}
		}		
	}	 
	
	/**
	 * returns the item in stored at the specified index.
	 */
	
	public <T> T returnStored(int index) throws IOException, ClassNotFoundException, IllegalAccessException,InstantiationException{
		readAvroFile();
		Object item = stored[index];
		String name = item.getClass().getName();
		if (name == "org.apache.avro.generic.GenericData$Array"){

			ArrayList arr = new ArrayList((List)stored[index]);

			scala_arr d = new scala_arr(arr);
			return (T)d;
		}
		else{
			return (T)stored[index];
		}

	}
	
	/**
	 * returns the whole array stored
	 */
	
	public Object[] returnStored() throws IOException, ClassNotFoundException, IllegalAccessException,InstantiationException{
		readAvroFile();
		return stored;
	}
	
	/**
	 * Only use if a List subclass (i.e. GenericData.Array) is in stored[index].
	 * Converts the List subclass to an array of the type of the example.
	 * NOTE: does not recursively convert. i.e. a list of lists will be converted
	 * to an array of lists. 
	 */
	
	public <T> T[] returnStoredArray(int index, T[] example){
		return (T[])((List)(stored[index])).toArray(example);
	}

		
	public void printStored(){
		System.out.println("begin printing args");
		for (Object a: stored){
			if (a == null){
				System.out.println("THE ARG... is null");
			}
			else {
				System.out.println("THE ARG..." + a);
				System.out.println("ITS CLASS..." + a.getClass());
			}
		}		
		System.out.println("end printing args");		
	}
	
	
	public static void main(String[] args) throws IOException, IllegalAccessException, ClassNotFoundException, InstantiationException{
		
		
	}
}
