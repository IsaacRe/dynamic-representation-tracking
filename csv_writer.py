

class CSVWriter:
    """
    Simple class to handle saving of data to csv file
    """

    def __init__(self, filename, *dtypes, open_now=False):
        self.file = filename
        self.data_buffer = {}
        self.dtypes = dtypes
        if open_now:
            self.open()
        else:
            self.stream = None
        self.lines = 0

    def open(self):
        self.stream = open(self.file, 'w+')
        self._write_header()

    def _write_header(self):
        line = ', '.join(self.dtypes)
        self.stream.write(line + '\n')
        self.stream.flush()

    def iterate(self):
        """
        Write current step's data to output file stream
        :return: None
        """
        line = ', '.join([self.data_buffer[d] if d in self.data_buffer else '' for d in self.dtypes])
        self.stream.write(line + '\n')
        self.stream.flush()
        self.lines += 1
        self.data_buffer = {}

    def write(self, **data):
        """
        Save provided data to the data buffer
        :param data: dict() of data items referenced by their dtype. Data should be convertible to string via str()
        :return: None
        """
        for dtype, d in data.items():
            # only allow dtypes that were specified at creation
            assert dtype in self.dtypes, "Specified data type not in self.dtypes"
            self.data_buffer[dtype] = str(d)

    def close(self):
        if self.data_buffer:
            self.iterate()
        self.stream.close()

    def __enter__(self):
        if not self.stream:
            self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
