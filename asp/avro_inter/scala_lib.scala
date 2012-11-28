package javro	
import java.util.ArrayList

class scala_iter[A](arr: ArrayList[Object])extends Iterator[A]{
	var stored = arr;
	var index = 0;

	def hasNext():Boolean={
		if (this.index < this.stored.size()){
			return true
		}else{ return false}
	}

	def next(): A={
		this.index +=1
		return (this.stored.get(this.index-1)).asInstanceOf[A]
	}
}

class scala_arr[A](arr: ArrayList[Object]) extends Seq[A]{	
	var stored = arr;

	def apply(idx:Int):A ={
		return this.stored.get(idx).asInstanceOf[A]
	}

	def iterator():Iterator[A]={
		return new scala_iter[A](this.stored)
	}

	def length():Int = {
		return this.stored.size()
	}
}

