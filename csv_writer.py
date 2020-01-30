

class CSVWriter:
    """
    Simple class to handle saving of data to csv file
    """

    def __init__(self, filename, *dtypes, open_now=False):
        self.file = filename
        self.data_buffer = {d: [] for d in dtypes}
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
        line = ','.join(self.dtypes)
        self.stream.write(line + '\n')
        self.stream.flush()

    def _check_buffer_full(self, **new_data):
        """
        Checks whether the current data buffer is populated for all dtypes (excluding those for which data has just
        been added).
        :param new_data: dict() of data that has just been added. If not specified all buffers must be populated
        :return: True if all buffers are full, else False
        """
        for dtype, data in self.data_buffer.items():
            if dtype in new_data:
                continue
            if len(data) == 0:
                return False
        return True

    def _collect(self, **new_data):
        """
        Collects the data at the front of each dtype's buffer. If buffer is empty use the newly added data.
        :param new_data: dict() containing the newly added data
        :return: list containing data for each dtype respecting ordering of self.dtypes
        """
        collect = []
        for dtype, buff in self.data_buffer.items():
            if len(buff) == 0:
                assert dtype in new_data, "Internal Failure: no data for dtype, %d, found in buffer or new_data" % dtype
                collect += [new_data[dtype]]
            else:
                collect += [buff.pop(0)]
        return collect

    def write(self, **data):
        """
        Save provided data to the data buffer
        :param data: dict() of data items referenced by their dtype. Data should be convertible to string via str()
        :return: True if writing to the outfile occurred, False if data was stored in memory
        """
        # convert to string
        data = {k: str(v) for k, v in data.items()}

        # check if the provided data completes the data buffer and we can write
        if self._check_buffer_full(**data):
            line = ', '.join(self._collect(**data))
            self.stream.write(line + '\n')
            self.stream.flush()
            self.lines += 1
            return True
        for dtype, d in data.items():
            # only allow dtypes that were specified at creation
            assert dtype in self.dtypes, "Specified data type not in self.dtypes"
            self.data_buffer[dtype] += [d]
        return False

    def close(self):
        self.stream.close()

    def __enter__(self):
        if not self.stream:
            self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
