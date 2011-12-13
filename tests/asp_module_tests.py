import unittest2 as unittest
import asp.jit.asp_module as asp_module
import asp.codegen.cpp_ast as cpp_ast 
from mock import Mock

class TimerTest(unittest.TestCase):
    def test_timer(self):
        pass
#         mod = asp_module.ASPModule()
#         mod.add_function("void test(){;;;;}", "test")
# #        mod.test()
#         self.failUnless("test" in mod.times.keys())
       
class ASPDBTests(unittest.TestCase):
    def test_creating_db(self):
        db = asp_module.ASPDB("test_specializer")

    def test_create_db_if_nonexistent(self):
        db = asp_module.ASPDB("test")
        self.assertTrue(db.connection)
    
    def test_create_table(self):
        db = asp_module.ASPDB("test")
        db.close() # close the real connection so we can mock it out
        db.connection = Mock()
        db.create_specializer_table()

        db.connection.execute.assert_called_with(
            'create table test (fname text, variant text, key text, perf real)')

    def test_insert(self):
        db = asp_module.ASPDB("test")
        db.close() # close the real connection so we can mock it out
        db.connection = Mock()
        db.table_exists = Mock(return_value = True)
        db.create_specializer_table()

        db.insert("func", "func", "KEY", 4.321)

        db.connection.execute.assert_called_with(
                'insert into test values (?,?,?,?)', ("func", "func", "KEY", 4.321))

    def test_create_if_insert_into_nonexistent_table(self):
        db = asp_module.ASPDB("test")
        db.close() # close the real connection so we can mock it out
        db.connection = Mock()

        # this is kind of a complicated situation.  we want the cursor to
        # return an array when fetchall() is called on it, and we want this
        # cursor to be created when the mock connection is asked for a cursor

        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = []
        db.connection.cursor.return_value = mock_cursor
        db.create_specializer_table = Mock()

        db.insert("func", "v1", "KEY", 4.321)

        self.assertTrue(db.create_specializer_table.called)

    def test_get(self):
        db = asp_module.ASPDB("test")
        db.close() # close the real connection so we can mock it out
        db.connection = Mock()
        db.table_exists = Mock(return_value = True)
        db.create_specializer_table()

        # see note about mocks in test_create_if...

        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = ['hello']
        db.connection.cursor.return_value = mock_cursor
        db.create_specializer_table = Mock()

        db.get("func")

        mock_cursor.execute.assert_called_with("select * from test where fname=?",
            ("func",))

    def test_update(self):
        db = asp_module.ASPDB("test")
        db.close() # close the real connection so we can mock it out
        db.connection = Mock()
        db.table_exists = Mock(return_value = True)
        
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = (1,)
        db.connection.cursor.return_value = mock_cursor


        db.update("foo", "foo_v1", "KEY", 3.21)

        db.connection.execute.assert_called_with("update test set perf=? where fname=? and variant=? and key=?",
                                               (3.21, "foo", "foo_v1", "KEY"))

    def test_delete(self):
        db = asp_module.ASPDB("test")
        db.close() # close the real connection so we can mock it out
        db.connection = Mock()
        db.table_exists = Mock(return_value = True)

        db.delete("foo", "foo_v1", "KEY")

        db.connection.execute.assert_called_with("delete from test where fname=? and variant=? and key=?",
                                                 ("foo", "foo_v1", "KEY"))






class SpecializedFunctionTests(unittest.TestCase):
        

    def test_creating(self):
        a = asp_module.SpecializedFunction("foo", None, Mock())

    def test_add_variant(self):        
        mock_backend = asp_module.ASPBackend(Mock(), None, Mock())
        a = asp_module.SpecializedFunction("foo", mock_backend, Mock())
        a.add_variant("foo_1", "void foo_1(){return;}")
        self.assertEqual(a.variant_names[0], "foo_1")
        self.assertEqual(len(a.variant_funcs), 1)

        # also check to make sure the backend added the function
        self.assertTrue(mock_backend.module.add_to_module.called)

        self.assertRaises(Exception, a.add_variant, "foo_1", None)

    def test_add_variant_at_instantiation(self):
        mock_backend = asp_module.ASPBackend(Mock(), None, Mock())
        a = asp_module.SpecializedFunction("foo", mock_backend, Mock(),
                                           ["foo_1"], ["void foo_1(){return;}"])
        self.assertEqual(len(a.variant_funcs), 1)
        self.assertTrue(mock_backend.module.add_to_module.called)

    def test_call(self):
        # this is a complicated situation.  we want the backend to have a fake
        # module, and that fake module should return a fake compiled module.
        # we'll cheat by just returning itself.
        mock_backend_module = Mock()
        mock_backend_module.compile.return_value = mock_backend_module
        mock_backend = asp_module.ASPBackend(mock_backend_module, None, Mock())
        mock_db = Mock()
        mock_db.get.return_value = []
        a = asp_module.SpecializedFunction("foo", mock_backend, mock_db)
        a.add_variant("foo_1", "void foo_1(){return;}")
        # test a call
        a()

        # it should call foo() on the backend module
        self.assertTrue(mock_backend_module.foo_1.called)

    def test_calling_with_multiple_variants(self):
        # this is a complicated situation.  we want the backend to have a fake
        # module, and that fake module should return a fake compiled module.
        # we'll cheat by just returning itself.
        mock_backend_module = Mock()
        mock_backend_module.compile.return_value = mock_backend_module
        mock_backend = asp_module.ASPBackend(mock_backend_module, None, Mock())
        mock_db = Mock()
        mock_db.get.return_value = []

        a = asp_module.SpecializedFunction("foo", mock_backend, mock_db)
        a.add_variant("foo_1", "void foo_1(){return;}")
        a.add_variant("foo_2", "void foo_2(){}")
        
        # test 2 calls
        a()
        # ensure the second one sees that foo_1 was called the first time
        mock_db.get.return_value = [["foo", "foo_1", None, None]]
        a()

        # it should call both variants on the backend module
        self.assertTrue(mock_backend_module.foo_1.called)
        self.assertTrue(mock_backend_module.foo_2.called)

    def test_pick_next_variant(self):
        mock_db = Mock()
        mock_db.get.return_value = []
        a = asp_module.SpecializedFunction("foo", Mock(), mock_db)
        a.add_variant("foo_1", "void foo_1(){return;}")
        a.add_variant("foo_2", "void foo_2(){}")

        self.assertEqual(a.pick_next_variant(), "foo_1")

        # now if one has run
        mock_db.get.return_value = [[None, "foo_1", None, None]]
        self.assertEqual(a.pick_next_variant(), "foo_2")

        # now if both have run
        mock_db.get.return_value = [[None, "foo_1", None, 1.0],
                                    [None, "foo_2", None, 2.0]]

        self.assertEqual(a.pick_next_variant(), "foo_1")




class HelperFunctionTests(unittest.TestCase):
    def test_creating(self):
        f = asp_module.HelperFunction("foo", "void foo(){}", Mock())

    def test_call(self):
        # this is a complicated situation.  we want the backend to have a fake
        # module, and that fake module should return a fake compiled module.
        # we'll cheat by just returning itself.
        mock_backend_module = Mock()
        mock_backend_module.compile.return_value = mock_backend_module
        mock_backend = asp_module.ASPBackend(mock_backend_module, None, Mock())
        a = asp_module.HelperFunction("foo", "void foo(){}", mock_backend)
        # test a call
        a()

        # it should call foo() on the backend module
        self.assertTrue(mock_backend_module.foo.called)


class ASPModuleMiscTests(unittest.TestCase):
    def test_generate(self):
        a = asp_module.ASPModule()
        mock_backend = Mock()
        a.backends["c++"] = mock_backend

        a.generate()

        self.assertTrue(mock_backend.module.generate.called)


        

class SingleFuncTests(unittest.TestCase):
    def test_adding_function(self):
        m = asp_module.ASPModule()
        m.add_function("foo", "void foo(){return;}")

        self.assertTrue(isinstance(m.specialized_functions["foo"],
                                   asp_module.SpecializedFunction))

    def test_adding_and_calling(self):
        m = asp_module.ASPModule()
        m.add_function("foo", "PyObject* foo(){Py_RETURN_TRUE;}")
        self.assertTrue(m.foo())

    def test_db_integration(self):
        m = asp_module.ASPModule()
        m.add_function("foo", "void foo(){return;}")
        m.foo()

        # Now let's check the db for what's inside
        self.assertEqual(len(m.db.get("foo")), 1)

    def test_helper_function(self):
        m = asp_module.ASPModule()
        m.add_helper_function("foo_helper", "PyObject* foo_helper(){Py_RETURN_TRUE;}")

        self.assertTrue("foo_helper" in m.specialized_functions)

        self.assertTrue(m.foo_helper())
         


class MultipleFuncTests(unittest.TestCase):
    def test_adding_multiple_variants(self):
        mod = asp_module.ASPModule()
        mod.add_function("foo", ["void foo_1(){};", "void foo_2(){};"],
                         ["foo_1", "foo_2"])
        self.assertTrue("foo_1" in mod.specialized_functions["foo"].variant_names)

    def test_running_multiple_variants(self):
        mod = asp_module.ASPModule()
        mod.add_function("foo", ["void foo_1(){printf(\"running foo1\\n\");};", 
                                 "void foo_2(){printf(\"running foo2\\n\");};"],
                         ["foo_1", "foo_2"])
        mod.foo()
        mod.foo()

        self.assertEqual(len(mod.db.get("foo")), 2)

    def test_running_variants_with_unrunnable_inputs(self):
        mod = asp_module.ASPModule()
        mod.add_function("foo", ["void foo_1(int a){};", "void foo_2(int a){};"],
                         ["foo_1", "foo_2"], [lambda *args,**kwargs: True, lambda *args,**kwargs: args[0] < 2])
        mod.foo(1)
        mod.foo(1)
        mod.foo(10)
        mod.foo(10)

        self.assertEqual(len(filter(lambda x: x[1]==u'foo_2',mod.db.get("foo"))),1)


"""


    def test_adding_multiple_versions(self):
        mod = asp_module.ASPModule()
        mod.add_function_with_variants(
            ["void test_1(){return;}", "void test_2(){return;}"],
            "test",
            ["test_1", "test_2"])
        mod.compile()
        self.failUnless("test" in mod.compiled_methods.keys())
        self.failUnless("test_1" in mod.compiled_methods["test"])

    def test_running_multiple_variants(self):
        mod = asp_module.ASPModule()
        mod.add_function_with_variants(
            ["PyObject* test_1(PyObject* a){return a;}", 
             "PyObject* test_2(PyObject* b){Py_RETURN_NONE;}"],
            "test",
            ["test_1", "test_2"])
        result1 = mod.test("a")
        result2 = mod.test("a")
        self.assertEqual(set([result1,result2]) == set(["a", None]), True)
        self.assertNotEqual(
            mod.compiled_methods["test"].database.get_oracular_best("test"),
            False)
        
    def test_running_multiple_variants_and_inputs(self):
        mod = asp_module.ASPModule()
	key_func = lambda name, *args, **_: (name, args) 
        mod.add_function_with_variants(
            ["void test_1(PyObject* a, PyObject* b){ long c = PyInt_AS_LONG(a); for(; c > 0; c--) b = PyNumber_Add(b,a); }", 
             "void test_2(PyObject* a, PyObject* b){ long c = PyInt_AS_LONG(b); for(; c > 0; c--) a = PyNumber_Add(a,b); }"] ,
            "test",
            ["test_1", "test_2"],
            key_func )
        val = 2000000
        mod.test(1,val)
        mod.test(1,val)
        mod.test(val,1)
        self.assertNotEqual(
            mod.compiled_methods["test"].database.get_oracular_best(key_func("test",1,val)), # best time found for this input
            False)
        self.assertEqual(
            mod.compiled_methods["test"].database.get_oracular_best(key_func("test",7,7)), # this input never previously tried
            False)
        self.assertEqual(
            mod.compiled_methods["test"].database.get_oracular_best(key_func("test",val,1)), # only one variant timed for this input
            False)
        mod.test(val,1)
        self.assertNotEqual(
            mod.compiled_methods["test"].database.get_oracular_best(key_func("test",val,1)), # now both variants have been timed
            False)
        self.assertEqual(mod.compiled_methods["test"].database.get_oracular_best(key_func("test",1,val)), 'test_1')
        self.assertEqual(mod.compiled_methods["test"].database.get_oracular_best(key_func("test",val,1)), 'test_2')

    def test_adding_variants_incrementally(self):
        mod = asp_module.ASPModule()
	key_func = lambda name, *args, **_: (name, args) 
        mod.add_function_with_variants(
            ["PyObject* test_1(PyObject* a, PyObject* b){ long c = PyInt_AS_LONG(a); for(; c > 0; c--) b = PyNumber_Add(b,a); return a;}"], 
            "test",
            ["test_1"],
            key_func )
        mod.test(1,20000)
        self.assertNotEqual(
            mod.compiled_methods["test"].database.get_oracular_best(key_func("test",1,20000)), # best time found for this input
            False)
        mod.add_function_with_variants(
             ["PyObject* test_2(PyObject* a, PyObject* b){ long c = PyInt_AS_LONG(b); for(; c > 0; c--) a = PyNumber_Add(a,b); return b;}"] ,
            "test",
            ["test_2"] )
        self.assertEqual(
            mod.compiled_methods["test"].database.get_oracular_best(key_func("test",1,20000)), # time is no longer definitely best
            False)
        mod.test(1,20000)
        mod.test(1,20000)
        self.assertNotEqual(
            mod.compiled_methods["test"].database.get_oracular_best(key_func("test",1,20000)), # best time found again
            False)
        self.assertEqual(mod.compiled_methods["test"].database.get_oracular_best(key_func("test",1,20000)), 'test_1')

    def test_pickling_variants_data(self):
        mod = asp_module.ASPModule()
	key_func = lambda name, *args, **_: (name, args) 
        mod.add_function_with_variants(
            ["PyObject* test_1(PyObject* a, PyObject* b){ long c = PyInt_AS_LONG(a); for(; c > 0; c--) b = PyNumber_Add(b,a); return a;}", 
             "PyObject* test_2(PyObject* a, PyObject* b){ long c = PyInt_AS_LONG(b); for(; c > 0; c--) a = PyNumber_Add(a,b); return b;}"] ,
            "test",
            ["test_1", "test_2"],
            key_func )
        mod.test(1,2)
        mod.test(1,2)
        mod.test(2,1)
        mod.save_method_timings("test")
        mod.clear_method_timings("test")
        mod.restore_method_timings("test")
        self.assertNotEqual(
            mod.compiled_methods["test"].database.variant_times[key_func("test",1,2)], # time found for this input
            False)
        self.assertEqual(
            key_func("test",7,7) not in mod.compiled_methods["test"].database.variant_times, # this input never previously tried
            True)
        self.assertEqual(
            len(mod.compiled_methods["test"].database.variant_times[key_func("test",2,1)]), # only one variant timed for this input
            1)

    def test_dealing_with_preidentified_compilation_errors(self):
        mod = asp_module.ASPModule()
        key_func = lambda name, *args, **_: (name, args)
        mod.add_function_with_variants(
            ["PyObject* test_1(PyObject* a, PyObject* b){ long c = PyInt_AS_LONG(a); for(; c > 0; c--) b = PyNumber_Add(b,a); return a;}", 
             "PyObject* test_2(PyObject* a, PyObject* b){ /*Dummy*/}",
             "PyObject* test_3(PyObject* a, PyObject* b){ long c = PyInt_AS_LONG(b); for(; c > 0; c--) a = PyNumber_Add(a,b); return b;}"] ,
            "test",
            ["test_1", "test_2", "test_3"],
            key_func,
            [lambda name, *args, **kwargs: True]*3,
            [True, False, True],
            ['a', 'b'] )
        mod.test(1,20000)
        mod.test(1,20000)
        mod.test(1,20000)
        self.assertNotEqual(
            mod.compiled_methods["test"].database.get_oracular_best(key_func("test",1,20000)), # best time found for this input
            False)
        self.assertEqual(
            mod.compiled_methods["test"].database.variant_times[("test",(1,20000))]['test_2'], # second variant was uncompilable
            -1)

    # Disabled, currently failing
    ""
    def test_dealing_with_preidentified_runtime_errors(self):
        mod = asp_module.ASPModule()
        key_func = lambda name, *args, **_: (name, args)
        mod.add_function_with_variants(
            ["PyObject* test_1(PyObject* a, PyObject* b){ long c = PyInt_AS_LONG(a); for(; c > 0; c--) b = PyNumber_Add(b,a); return a;}", 
             "PyObject* test_2(PyObject* a, PyObject* b){ long c = PyInt_AS_LONG(a); for(; c > 0; c--) b = PyNumber_Add(b,a); return a;}", 
             "PyObject* test_3(PyObject* a, PyObject* b){ long c = PyInt_AS_LONG(b); for(; c > 0; c--) a = PyNumber_Add(a,b); return b;}"] ,
            "test",
            ["test_1", "test_2", "test_3"],
            key_func,
            [lambda name, *args, **kwargs: True, lambda name, *args, **kwargs: args[1] < 10001, lambda name, *args, **kwargs: True],
            [True]*3,
            ['a', 'b'] )
        result1 = mod.test(1,20000)
        result2 = mod.test(1,20000)
        result3 = mod.test(1,20000)
        result1 = mod.test(1,10000)
        result2 = mod.test(1,10000)
        result3 = mod.test(1,10000)
        self.assertNotEqual(
            mod.compiled_methods["test"].database.get_oracular_best(key_func("test",1,20000)), # best time found for this input
            False)
        self.assertNotEqual(
            mod.compiled_methods["test"].database.get_oracular_best(key_func("test",1,10000)), # best time found for this input
            False)
        self.assertEqual(
            mod.compiled_methods["test"].database.variant_times[("test",(1,20000))]['test_2'], # second variant was unrannable for 20000
            -1)
        self.assertNotEqual(
            mod.compiled_methods["test"].database.variant_times[("test",(1,10000))]['test_2'], # second variant was runnable for 10000
            -1)
    """

if __name__ == '__main__':
    unittest.main()
