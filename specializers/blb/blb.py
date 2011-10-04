class BLB(object):
    
    def __init__(self):
        pass

    def run(self, arr):
        if not hasattr(self, 'compute_estimates') or not hasattr(self, 'reduce_bootstraps') or \
           not hasattr(self, 'average'):
           raise ArgumentException("Undefined methods")

        import asp.codegen.templating.template as template
        mytemplate = template.Template(filename="templates/BLB.mako", disable_unicode=True)
        rendered = mytemplate.render(computeEstimate=self.compute_estimates,
                                     reduceBootstraps=self.reduce_bootstraps,
                                     average=self.average)

        import asp.jit.asp_module as asp_module
        mod = asp_module.ASPModule(use_scala=True)
        # remember, must specify function name when using a string
        mod.add_function("BLB", rendered, backend='scala')
        return mod.BLB(arr)

