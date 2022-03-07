import bisect
import collections.abc
import math
import numbers

class Schedule:
    """
    This object acts as a container and interface for modifying the agent's schedule.
    """
    def __init__(self, associated_agent):
        # reference objects
        self._agent = associated_agent
        self._current_time = self._agent.env.timestamp
        self._current_activity = self._agent.mobility_planner.current_activity

        # load schedule object
        self.schedule = RangeDict()
        self.load_current_day()

    def load_current_day(self):
        """
        pull the current day's schedule into our representation
        """
        # get current_state
        current_state = self._current_activity.clone()
        start = current_state.start_time
        end = current_state.end_time
        self.schedule[(start, end)] = current_state
        # get the activities that are queued for every day except the first day.
        for activity in self._agent.mobility_planner.schedule_for_day:
            start = activity.start_time
            end = activity.end_time
            self.schedule[(start, end)] = activity

    # =========================================
    # checks and balances on a given adjustment
    # =========================================
    def _is_current(self, activity):
        """
        is this the current activity based on the time.
        Args:
            activity: a given activity object (tested to see if it is the current object)
        Returns:
            False if this is not the current activity, True if it is the current activity
        """
        return activity.start_time <= self._current_time < activity.end_time

    def _in_future(self, activity):
        """
        is the chosen operation in the future based on the current time.
        Args:
            activity: given activity object
        Returns:
            True if the activity hasnt started yet. False if it has.
        """
        return activity.start_time > self._current_time

    def _is_sleep(self, activity):
        """
        I am making it a hard and fast rule that sleep cannot be edited. This is because
        we initialize the simulation with an tricky sleep state. To avoid very messy desyncs
        between this API and the main simulation state, I have opted to forgo sleep edits.
        After all, digital agents need sleep too.
        Args:
            activity: the activity to be examined.
        Returns:
            True if the activity is sleep.
        """
        return activity.name == 'sleep'

    # this is a check that should be made before any edit is applied
    def can_edit(self, activity):
        """
        Checks if an edit can be made to the activity based on my 3 constraints
        1. Edits must not be to the current simulation.
        2. Edits must be to future scheduling.
        3. Edits cannot adjust sleep.
        Args:
            activity: given activity object
        Returns:
            True if all my constraints are met
        """
        return not self._is_current(activity) and self._in_future(activity) and not self._is_sleep(activity)

    # ====================================
    # activity editing API
    # ====================================
    def cancel_activity(self, timestamp):
        """
        cancels the given activity, and sets it to be at home during that time.
        returns True on Success.
        True on Success, False on Failure.
        """
        activity = self.schedule[timestamp]
        if self._can_edit(activity):
            activity.cancel_and_go_to_location("MANUAL OVERRIDE")
            return True
        return False

    def delete_activity(self, timestamp):
        """
        Takes the timestamp of an activity, and deletes it from the queue.
        Replaces it with idle time. If idle time is next to this object, then
        stretch that instead. (cannot delete sleep).
        Args:
            timestamp: time of activity
        Returns:
            True on success, False on failure.
        """
        activity = self.schedule.get(timestamp)
        if not self.can_edit(activity):
            return

        next_activity = self.schedule.get(activity.end_time)
        # get activity,
        # get next activity
        # if next activity is idle; remove this from queue, and extend into space.
        # else; remove from queue, and insert new idle time.
        # todo delete activity if possible.

    def insert_activity(self, start_timestamp, activity_type, end_timestamp):
        """
        create a new activity object, and push it into today, both on our map and in the actual queue.
        Args:
            start_timestamp: what time does it start
            activity_type: type of activity.
            end_timestamp: what time does it end.

        Returns:
            True on success, False on failure
        """
        # todo insert a new activity if possible

    def adjust_start_time(self, timestamp, new_start_time):
        """
        push the start time up a little bit (here and in actual planner),
        but check that it doesn't impinge upon anything else first.
        Args:
            timestamp: time of activity that you want to edit.
            new_start_time: time you want to move its start time to.

        Returns:
            True on success, False on failure
        """
        # todo move start time if no conflicts, otherwise moves as far as possible in given direction

    def adjust_end_time(self, timestamp, new_start_time):
        """
        push the end time up a little bit (here and in actual planner),
        but check that it doesn't impinge upon anything else first.
        Args:
            timestamp: time of activity that you want to edit.
            new_start_time: time you want to move its start time to.

        Returns:
            True on success, False on failure
        """
        # todo move end time if no conflicts, otherwise moves as far as possible in given direction

    def relocate_activity(self, timestamp, new_location):
        """
        move a future event somewhere else.
        Args:
            timestamp: time of activity you want to edit.
            new_location: reference to a different location.

        Returns:
            True on success, False on failure
        """
        # todo move events if in future.

# =================
# ranged dict class
# =================
def _binary_key_search(LUT, key):
    """
    binary search through the LUT, assumes forward LUT.
    """
    Lower = 0
    Upper = len(LUT)
    while Lower <= Upper:
        i = math.floor((Lower + Upper) / 2)  # get middle index
        if i == len(LUT):
            break
        if LUT[i][1] <= key:
            Lower = i + 1
        elif LUT[i][0] > key:
            Upper = i - 1
        elif LUT[i][0] <= key < LUT[i][1]:
            return i
    raise RangDictKeyException("Key not found")


class RangeDict(collections.abc.MutableMapping):
    """
    This datastructure takes a minimum and max as a key, and maps it to a given value.
    """

    def __init__(self, *args, **kwargs):
        self._store = dict()
        self._FLUT = []  # key (start, end)
        self._BLUT = []  # key (end, start); used purely to double check that the given addition is valid, inherent override rejection.
        self.update(dict(*args, **kwargs))

    def __getitem__(self, key: numbers.Number or (numbers.Number, numbers.Number)):
        # grab the nearest range. I this should be O(log n)
        if key is tuple and key in self._store:
            return self._store[key]

        i = _binary_key_search(self._FLUT, key)
        bounds = self._FLUT[i]
        # test that we are in range.
        return self._store[self._FLUT[i]]

    def __setitem__(self, key: (numbers.Number, numbers.Number), value):
        # (O(n)) insert step, I wish it were faster, but to get fast lookup takes time.
        self._check_range(key)
        bisect.insort_left(self._FLUT, (key[0], key[1]))
        bisect.insort_left(self._BLUT, (key[1], key[0]))
        self._store[key] = value

    def __delitem__(self, key):
        # again, time complexity O(n), the random access really hurts us here.
        i = _binary_key_search(self._FLUT, key)
        del self._store[self._FLUT[i]]
        del self._FLUT[i]
        del self._BLUT[i]

    def __iter__(self):
        return iter(self._store)

    def __len__(self):
        return len(self._store)

    def _check_range(self, key: (numbers.Number, numbers.Number)):
        '''
        attempt to assign the key, can fail if range is of bad format, or if overlaps with an existing range
        '''
        if key[0] >= key[1]:
            raise RangDictKeyException("Range is not valid, should be (min, max); must be different")

        # forward check
        i_f = bisect.bisect_left(self._FLUT, (key[0], key[0]))
        i_b = bisect.bisect_left(self._BLUT, (key[1], key[1]))
        # test that we have empty lookup tables (initializing table)
        if not self._FLUT and not self._BLUT:
            return
        # test that the relative indices match (should work for all LUT size > 0)
        if i_f == i_b:
            # case: at the start
            if i_f == 0 and self._FLUT[i_f][0] < key[1]:
                raise RangDictKeyException(
                    "Range is not unique (ending of new block impinges upon existing start of a block).")
            # case: at the end
            elif i_f == len(self._FLUT) and self._FLUT[-1][1] > key[0]:
                raise RangDictKeyException("Range is not unique (start of new block is within the previous block).")
            # case: at the middle
            elif (i_f != len(self._FLUT) and i_f != 0) and (
                    self._FLUT[i_f - 1][1] > key[0] or self._FLUT[i_f][0] < key[1]):
                raise RangDictKeyException("Range is not unique (block is contained in different block).")
            # all tests passed.
            else:
                return
        raise RangDictKeyException("LUT index mismatch (should remain in sync).")


class RangDictKeyException(Exception):
    pass
