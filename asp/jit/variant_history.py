
class CodeVariantPerformanceDatabase(object):
    def __init__(self):
        self.variant_times = {}    
           # The measured performance data for a particular method
           # Dict of dicts, key: input key, value: dict  
           # Inner dict of times, key: v_id, value: time 
           # The set of v_ids contained in the dict may be 
           # much larger than the set of v_ids compiled by
           # a particular instance of a specializer
        self.oracular_best = {} 
           # The variant id of the best variant out of all 
           # currently compiled variants of a particular method
           # Dict of v_ids, key: input key, value: v_id or False

    def set_oracular_best(self, key, time_dict, v_id_set):
        # filter entries that failed to run or are not currently compiled
        succeeded = filter( lambda x: x[1] > 0 and x[0] in v_id_set, \
                            time_dict.iteritems() ) 
        if not succeeded: 
            print "Warning: ASP has tried every currently compiled variant for this input and none have run successfully. Add different variants."
            self.oracular_best[key] = False
        else: 
            name = min(succeeded, key=lambda p: p[1])[0] # key with min val
            self.oracular_best[key] = name

    def get_oracular_best(self, key):
        return self.oracular_best.get(key, False)

    def clear_oracle(self):
        self.oracular_best.clear() #newly added variant might be the best

    def add_time(self, key, elapsed, v_id, v_id_set):
        time_dict = self.variant_times.get(key,{})
        # TODO: Overwrite old times with new data? If so, reset best when?
        if v_id not in time_dict:
            time_dict[v_id] = elapsed
            if set(time_dict.keys()) >= v_id_set:
                self.set_oracular_best(key, time_dict, v_id_set)
            self.variant_times[key] = time_dict

    def get_measured_v_ids(self, key):
        return self.variant_times.get(key,{}).keys()

    def clear(self):
        self.variant_times.clear()
        self.oracular_best.clear()

    def get_picklable_obj(self):
        return { 'variant_times': self.variant_times }

    def set_from_pickled_obj(self, obj, v_id_set):
        self.variant_times = obj['variant_times']
        for k, time_dict in self.variant_times.iteritems():
            if set(time_dict.keys()) >= v_id_set:
                self.set_oracular_best(k, time_dict, v_id_set)


class CodeVariantUseCaseLimiter(object):
    def __init__(self):
        self.compilable = {} 
            # Track whether or not a variant is compilable on this machine
            # Dict of bools, key: v_id, val: bool
        self.input_limits_funcs = {} 
            # Return a function determining if a particular input is 
            # runnable with a particular variant
            # Dict of closures, key: v_id, val: closure returning bool 

    def is_allowed(self, v_id, *args, **kwargs):
        return self.compilable[v_id] and  \
               self.input_limits_funcs[v_id](*args, **kwargs)

    def append(self, v_id_list, limit_funcs, compilables):
        for v, lim, c in zip(v_id_list, limit_funcs, compilables):
            self.input_limits_funcs[v] = lim
            self.compilable[v] = c


class CodeVariantSelector(object):
    def __init__(self, perf_database, use_case_limiter):
        self.perf_database = perf_database
        self.use_case_limiter = use_case_limiter

    def get_v_id_to_run(self, v_id_set, key, *args, **kwargs):

        def exhaustive_search():
            candidates = v_id_set - set(self.perf_database.get_measured_v_ids(key))
            while candidates:
                v_id = candidates.pop()
                if self.use_case_limiter.is_allowed(v_id, *args, **kwargs):
                    return v_id
                self.perf_database.add_time(key, -1., v_id, v_id_set)
            return None

        best = self.perf_database.get_oracular_best(key)
        return best if best else exhaustive_search()
        return ret_func or error_func

    def use_supplied_function_to_generate_a_new_variant():
        pass

class CodeVariants(object):
    def __init__(self, variant_names, key_func, param_names):
        self.v_id_list = variant_names
        self.v_id_set = set(variant_names)
        self.make_key = key_func     
        self.param_names = param_names
        self.database = CodeVariantPerformanceDatabase()
        self.limiter = CodeVariantUseCaseLimiter()
        self.selector = CodeVariantSelector(self.database, self.limiter)

    def __contains__(self, v_id):
        return v_id in self.v_id_list

    def append(self, variant_names):
        self.v_id_list.extend(variant_names)
        self.v_id_set.update(variant_names)

    def get_picklable_obj(self):
        return {
                'variant_names': self.v_id_list,
                'param_names': self.param_names,
               }

    def set_from_pickled_obj(self, obj):
        if self.v_id_list != obj['variant_names']:
	    print "Warning: Attempted to load pickled performance data for non-matching space of code variants."
	    return
        self.param_names = obj['param_names']

