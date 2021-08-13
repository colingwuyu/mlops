import os


class FileLineCounter(object):
    """FileLineCounter
    Counter tracking file line numbers

    Increment counter by pre-defined inc_interval when file # lines satisfy the interval

    """

    def __init__(
        self,
        data_file="data.csv",
        headers=True,
        inc_interval=100,
        counter_file="counter",
        initial_point=0,
    ):
        """constructor

        Args:
            data_file (str, optional): File name it tracks. Defaults to "data.csv".
            headers (bool, optional): File contains header line or not. Defaults to True.
            inc_interval (int, optional): Number of lines for incremental interval. Defaults to 100.
            counter_file (str, optional): Counter log file name. Defaults to "counter".
            initial_point (int, optional): Initial point counter starts tracking on. Default to 0
        """
        self._cur_count = None
        # self._file_lines = None
        self._inc_interval = inc_interval
        self._data_file = data_file
        self._headers = headers
        self._counter_file = counter_file
        self._initial_point = initial_point
        self.load()

    @property
    def file_lines(self):
        return self.count_lines()

    @property
    def inc_interval(self):
        return self._inc_interval

    def save(self):
        """Persistent counter"""
        with open(self._counter_file, "w") as f:
            f.write(str(self._cur_count))

    def load(self):
        """Initialize counter"""
        if not os.path.exists(self._counter_file):
            self._cur_count = self._initial_point
            self.save()
        else:
            with open(self._counter_file, "r") as f:
                self._cur_count = int(f.read())
        # self._file_lines = self.count_lines()

    def clear(self):
        os.remove(self._counter_file)
        self.load()

    def move(self):
        """generator yields next batch lines range [start_line, end_line] (inclusive)

        Yields:
            tuple(int): a range for an increment inclusive ([start_line, end_line])
        """
        while self.can_move():
            start_line = self._cur_count + 1
            self.inc()
            end_line = self._cur_count
            yield start_line, end_line

    def move_to(self, line_num):
        """move to the given line directly"""
        self._cur_count = line_num
        self.save()

    def inc(self):
        self._cur_count += self._inc_interval
        self.save()

    def can_move(self):
        count = self.count_lines()
        return count - self._cur_count >= self._inc_interval

    def count_lines(self):
        file = open(self._data_file, "r")
        nonempty_lines = [line.strip("\n") for line in file if line != "\n"]
        line_count = len(nonempty_lines)
        file.close()
        if self._headers:
            return line_count - 1
        else:
            return line_count
