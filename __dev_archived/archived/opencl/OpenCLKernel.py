class OpenCLKernel:
    """
    OpenCL kernel.

    It does not allocate any resources, thus can be used as static class variable.

    arguments

        kernel_text    OpenCL text of kernel. Must contain only one kernel.

        global_shape    default global_shape for .run()
        
        local_shape     default local_shape for .run()
    """
    def __init__(self, kernel_text, global_shape=None, local_shape=None):
        self.kernel_text = kernel_text
        self.global_shape = global_shape
        self.local_shape = local_shape

    def __str__(self):  return f'OpenCLKernel ({self.kernel_text})'
    def __repr__(self): return self.__str__()