import simpy
import datetime
from orderedset import OrderedSet
from collections import namedtuple
import numpy as np
import warnings

from covid19sim.utils.constants import SECONDS_PER_MINUTE, SECONDS_PER_HOUR, AGE_BIN_WIDTH_5, ALL_LOCATIONS, WEEKDAYS, ALL_DAYS, SECONDS_PER_DAY
from covid19sim.epidemiology.p_infection import get_environment_human_p_transmission
from covid19sim.epidemiology.viral_load import compute_covid_properties
from covid19sim.log.event import Event

DistanceProfile = namedtuple("DistanceProfile", ['encounter_term', 'social_distancing_term', 'packing_term', 'distance'])
class Location(simpy.Resource):
    """
    Class representing generic locations used in the simulator
    """

    def __init__(self, env, rng, conf, area, name, location_type, lat, lon, capacity):
        """
        Locations are created with city.create_location(), not instantiated directly

        Args:
            env (covid19sim.Env): Shared environment
            rng (np.random.RandomState): Random number generator
            area (float): Area of the location
            name (str): The location's name
            type (str): Location's type, see
            lat (float): Location's latitude
            lon (float): Location's longitude
            social_contact_factor (float): how much people are close to each other
                see contamination_probability() (this scales the contamination pbty)
            capacity (int): Daily intake capacity for the location (infinity if None).
            surface_prob (float): distribution of surface types in the Location. As
                different surface types have different contamination probabilities
                and virus "survival" durations, this will influence the contamination
                of humans at this location.
                Surfaces: aerosol, copper, cardboard, steel, plastic
        """

        assert location_type in ALL_LOCATIONS, "not a valid location"
        if capacity is None:
            capacity = simpy.core.Infinity

        super().__init__(env, capacity)
        self.humans = OrderedSet()  # OrderedSet instead of set for determinism when iterating
        self.conf = conf
        self.name = name
        self.rng = np.random.RandomState(rng.randint(2 ** 16))
        self.lat = lat
        self.lon = lon
        self.area = area
        self.location_type = location_type
        self.env = env
        self.contamination_timestamp = datetime.datetime.min
        self.max_day_contamination = 0
        self.is_open_for_business = True
        self.binned_humans = {bin:OrderedSet() for bin in AGE_BIN_WIDTH_5}

        # occupation related constants
        OPEN_CLOSE_TIMES = conf[f'{location_type}_OPEN_CLOSE_HOUR_MINUTE']
        OPEN_DAYS = conf[f'{location_type}_OPEN_DAYS']
        # /!\ opening and closing time are in seconds relative to midnight
        self.opening_time = OPEN_CLOSE_TIMES[0][0] * SECONDS_PER_HOUR +  OPEN_CLOSE_TIMES[0][1] * SECONDS_PER_MINUTE
        self.closing_time = OPEN_CLOSE_TIMES[1][0] * SECONDS_PER_HOUR +  OPEN_CLOSE_TIMES[1][1] * SECONDS_PER_MINUTE
        self.open_days = OPEN_DAYS

        # parameters related to sampling contacts
        self.CONTACT_DURATION_GAMMA_SCALE_MATRIX = np.array(conf['CONTACT_DURATION_GAMMA_SCALE_MATRIX'])
        self.CONTACT_DURATION_GAMMA_SHAPE_MATRIX = np.array(conf['CONTACT_DURATION_GAMMA_SHAPE_MATRIX'])
        self.MEAN_DAILY_UNKNOWN_CONTACTS = conf['MEAN_DAILY_UNKNOWN_CONTACTS']

        # contact related constants
        self.social_contact_factor = conf[f'{location_type}_CONTACT_FACTOR']
        self.contaminated_surface_probability = conf[f'{location_type}_SURFACE_PROB']
        self.MEAN_DAILY_KNOWN_CONTACTS = conf[f'{location_type}_MEAN_DAILY_INTERACTIONS']

        if location_type in ["SENIOR_RESIDENCE", "HOUSEHOLD"]:
            key = "HOUSEHOLD"
        elif location_type == "SCHOOL":
            key = "SCHOOL"
        elif location_type == "WORKPLACE":
            key = "WORKPLACE"
        else:
            key = "OTHER"

        self.P_CONTACT = np.array(conf[f'P_CONTACT_MATRIX_{key}'])
        self.ADJUSTED_CONTACT_MATRIX = np.array(conf[f'ADJUSTED_CONTACT_MATRIX_{key}'])
        self.MEAN_DAILY_KNOWN_CONTACTS_FOR_AGEGROUP = self.ADJUSTED_CONTACT_MATRIX.sum(axis=0)

        for matrix in [self.CONTACT_DURATION_GAMMA_SCALE_MATRIX, self.CONTACT_DURATION_GAMMA_SHAPE_MATRIX, self.P_CONTACT, self.ADJUSTED_CONTACT_MATRIX ]:
            assert matrix.shape[0] == matrix.shape[1], "contact matrix is not square"

        # intervened contacts
        N_LEVELS = self.conf['N_LEVELS']

        # if hospitals are assumed to be safe, we reduce the number of interactions to 0
        if location_type == "HOSPITAL" and conf['ASSUME_SAFE_HOSPITAL_DAILY_INTERACTIONS']:
            self.DAILY_INTERACTION_REDUCTION_FACTOR = [0] * N_LEVELS

        # there are a total of N_LEVELS. Last one is known to have a reduction factor with respect to the unconfined scenario
        penultimate_level = self.conf[f"LOCKDOWN_FRACTION_REDUCTION_IN_CONTACTS_AT_{key}"]
        reduction_level = []
        for i in range(N_LEVELS + 1):
            if i == 0:
                x = 0
            elif i == N_LEVELS:
                x = 1
            elif i == N_LEVELS - 1:
                x = penultimate_level
            else:
                x = (N_LEVELS - 1 + i) * penultimate_level / (N_LEVELS - 1)
            reduction_level.append(x)

        self.DAILY_INTERACTION_REDUCTION_FACTOR = reduction_level

    def infectious_human(self):
        """
        Returns:
            bool: Is there an infectious human currently at that location
        """
        for h in self.humans:
            if h.is_infectious:
                return True
        return False

    def __repr__(self):
        """
        Returns:
            str: Representation of the Location
        """
        return f"{self.name} - occ:{len(self.humans)}/{self.capacity} - I:{self.is_contaminated}"

    def add_human(self, human):
        """
        Adds a human instance to the OrderedSet of humans at the location.

        Args:
            human (covid19sim.human.Human): The human to add.
        """
        self.humans.add(human)
        self.binned_humans[human.age_bin_width_5.bin].add(human)

    def remove_human(self, human):
        """
        Remove a given human from location.human
        If they are infectious, then location.contamination_timestamp is set to the
        env's timestamp and the duration of this contamination is set
        (location.max_day_contamination) according to the distribution of surfaces
        (location.contaminated_surface_probability) and the survival of the virus
        per surface type (MAX_DAYS_CONTAMINATION)
        /!\ Human is not returned

        Args:
            human (covid19sim.human.Human): The human to remove
        """
        if human in self.humans:
            if human.is_infectious:
                self.contamination_timestamp = self.env.timestamp
                rnd_surface = float(self.rng.choice(
                    a=human.conf.get("MAX_DAYS_CONTAMINATION"),
                    size=1,
                    p=self.contaminated_surface_probability
                ))
                self.max_day_contamination = max(self.max_day_contamination, rnd_surface)
            self.humans.remove(human)
            self.binned_humans[human.age_bin_width_5.bin].remove(human)

    @property
    def is_contaminated(self):
        """
        Is the location currently contaminated? It is if one of these two
        conditions is true :
        - an infectious human is present at the location (location is
          considered constantly reinfected by the human)
        - there are no infectious humans but the location was contaminated
          recently (see remove_human()). This depends on the time the last
          infectious human left, the current time and the duration of the
          contamination (location.max_day_contamination).

        Returns:
            bool: Is the place currently contaminating?
        """
        if self.infectious_human():
            return True
        else:
            return (self.env.timestamp - self.contamination_timestamp <=
                    datetime.timedelta(days=self.max_day_contamination))

    @property
    def contamination_probability(self):
        """
        Contamination depends on the time the virus has been sitting on a given surface
        (location.max_day_contamination) and is linearly decayed over time.
        Then it is scaled by location.social_contact_factor

        If not location.is_contaminated, return 0.0

        Returns:
            float: probability that a human is contaminated when going to this location.
        """
        if self.is_contaminated:
            if self.infectious_human():
                # Location constantly reinfected by the infectious human
                p_infection = 1.0
            else:
                # Linear decay of p_infection depending on time since last infection
                lag = (self.env.timestamp - self.contamination_timestamp)
                lag /= datetime.timedelta(days=1)
                p_infection = 1 - lag / self.max_day_contamination
        else:
            p_infection = 0.0

        return p_infection

    def _sample_interactee(self, type, human):
        """
        Samples encounter partner of `type` for `human`

        Args:
            type (string): type of interaction to sample. expects "known", "unknown"
            human (covid19sim.human.Human): human who will interact with the sampled human

        Returns:
            other_human (covid19sim.human.Human): `human` with whom this `human` will interact
        """
        if len(self.humans) == 1:
            return None

        PREFERENTIAL_ATTACHMENT_FACTOR = self.conf['PREFERENTIAL_ATTACHMENT_FACTOR']

        def _extract_attrs(human, candidate):
            return (
                candidate,
                candidate.age_bin_width_5.index,
                candidate in human.known_connections,
                _get_daily_interaction_reduction_factor(candidate)
            )

        if type == "known":
            human_bin = human.age_bin_width_5.index
            candidate_humans = [h for h in self.humans if human != h]
            hs, h_vector, known_vector, reduction_factor = list(zip(*[_extract_attrs(human, h) for h in candidate_humans]))

            p_contact = self.P_CONTACT[h_vector, human_bin] * (1 - PREFERENTIAL_ATTACHMENT_FACTOR)
            p_contact += np.array(known_vector) * PREFERENTIAL_ATTACHMENT_FACTOR
            # reduction factor is due to mutual interaction sampling where other_human's reduction factor is taken into account
            p_contact *= (1 - np.array(reduction_factor))
            if p_contact.sum() == 0:
                return None

            p_contact /= p_contact.sum()

            # sample
            other_human = self.rng.choice(hs, size=1, p=p_contact).item()

        elif type == "unknown":
            other_humans = [x for x in self.humans if x != human]
            other_human = self.rng.choice(other_humans, size=1).item()
        else:
            raise

        assert other_human != human, "interaction with oneself should not happen"
        return other_human

    def _sample_interaction_with_type(self, type, human):
        """
        Samples interactions of type `type` for `human`.

        Args:
            type (string): type of interaction to sample. expects "known", "unknown"
            human (covid19sim.human.Human): human who will interact with the sampled human

        Returns:
            interactions (list): each element is as follows -
                human (covid19sim.human.Human): other human with whom to have `type` of interaction
                distance_profile (covid19sim.locations.location.DistanceProfile): distance from which these two humans met (cms)
                duration (float): duration for which this encounter took place (seconds)
        """

        if type == "known":
            mean_daily_interactions = self.MEAN_DAILY_KNOWN_CONTACTS_FOR_AGEGROUP[human.age_bin_width_5.index]
            min_dist_encounter = self.conf['MIN_DIST_KNOWN_CONTACT']
            max_dist_encounter = self.conf['MAX_DIST_KNOWN_CONTACT']
            mean_interaction_time = None
        elif type == "unknown":
            mean_daily_interactions = self.MEAN_DAILY_UNKNOWN_CONTACTS
            min_dist_encounter = self.conf['MIN_DIST_UNKNOWN_CONTACT']
            max_dist_encounter = self.conf['MAX_DIST_UNKNOWN_CONTACT']
            mean_interaction_time = self.conf["GAMMA_UNKNOWN_CONTACT_DURATION"]
        else:
            raise

        mean_daily_interactions += 1e-6 # to avoid error in sampling with 0 mean from negative binomial
        mean_daily_interactions *= 1 - self.DAILY_INTERACTION_REDUCTION_FACTOR[human.behavior_level]
        scale_factor_interaction_time = self.conf['SCALE_FACTOR_CONTACT_DURATION']
        # (assumption) maximum allowable distance is when humans are uniformly spaced
        packing_term = 100 * np.sqrt(self.area/len(self.humans))

        interactions = []
        n_interactions = self.rng.negative_binomial(mean_daily_interactions, 0.5)
        for i in range(n_interactions):
            # sample other human
            other_human = self._sample_interactee(type, human)
            if other_human is None:
                continue

            # sample distance of encounter
            encounter_term = self.rng.uniform(min_dist_encounter, max_dist_encounter)
            social_distancing_term = np.mean([human.maintain_extra_distance, other_human.maintain_extra_distance]) #* self.rng.rand()
            distance = np.clip(encounter_term + social_distancing_term, a_min=0, a_max=packing_term)
            distance_profile = DistanceProfile(encounter_term=encounter_term, packing_term=packing_term, social_distancing_term=social_distancing_term, distance=distance)

            # sample duration of encounter (seconds)
            t_overlap = (min(human.location_leaving_time, other_human.location_leaving_time) -
                         max(human.location_start_time,   other_human.location_start_time))

            if type == "known":
                age_bin = human.age_bin_width_5.index
                other_bin = other_human.age_bin_width_5.index
                scale_duration = self.CONTACT_DURATION_GAMMA_SCALE_MATRIX[other_bin, age_bin]
                shape_duration = self.CONTACT_DURATION_GAMMA_SHAPE_MATRIX[other_bin, age_bin]
                # surveyed data gives us minutes per day. Here we use it to sample rate of minutes spend per second of overlap in an encounter.
                duration = (self.rng.gamma(shape_duration, scale_duration) / SECONDS_PER_DAY) * t_overlap * SECONDS_PER_MINUTE

            elif type == "unknown":
                duration = self.rng.gamma(mean_interaction_time/scale_factor_interaction_time, scale_factor_interaction_time)

            else:
                raise ValueError

            # if self.location_type ==  "MISC" and human.workplace != self:
            #     print(human, "-->", other_human, "for", duration, "tota humans", len(self.humans), "t_overlap", t_overlap, human.mobility_planner.current_activity)

            if self.location_type in ["MISC", "EXERCISE"]:
                duration *= 100
            # if duration > t_overlap:
            #     warnings.warn(f"sampled time {duration} is more than the duration for which {human} is at a location {t_overlap}")

            # /!\ clipping changes the distribution.
            duration = min(t_overlap, duration) * max(human.time_encounter_reduction_factor, other_human.time_encounter_reduction_factor)

            # add to the list
            interactions.append((other_human, distance_profile, duration))

        return interactions

    def sample_interactions(self, human, unknown_only=False):
        """
        samples how `human` interacts with other `human`s at this location (`self`) at this time.

        Args:
            human (covid19sim.human.Human): human for whom interactions need to be sampled
            unknown_only (bool): whether to sample interactions of type `unknown` only

        Returns:
            known_interactions (list): each element is as follows -
                human (covid19sim.human.Human): other human with whom to interact
                distance_profile (covid19sim.locations.location.DistanceProfile): distance from which this encounter took place (cms)
                duration (float): duration for which this encounter took place (minutes)
            unknown_interactions (list): each element is as follows -
                human (covid19sim.human.Human): other human who was nearby and unknown to `human`
                distance_profile (covid19sim.locations.location.DistanceProfile): distance from which this encounter took place (cms)
                duration (float): duration for which this encounter took place (minutes)
        """
        # only `human` is at this location. There will be no interactions.
        if len(self.humans) == 1:
            assert human == self.humans[0]
            return [], []

        known_interactions = []
        unknown_interactions = self._sample_interaction_with_type("unknown", human)
        if not unknown_only:
            known_interactions = self._sample_interaction_with_type("known", human)

        return known_interactions, unknown_interactions

    def check_environmental_infection(self, human):
        """
        Determines whether `human` gets infected due to the virus in environment.

        Environmental infection is modeled via surfaces. We consider the following study -
        https://www.nejm.org/doi/pdf/10.1056/NEJMc2004973?articleTools=true
        It shows the duration for which virus remains on a surface. Following surfaces are considered -
        aerosol    copper      cardboard       steel       plastic

        We sample a surface using surface_prob and infect the surface for MAX_DAYS_CONTAMINATION[surface_index] days.
        NOTE: self.surface_prob is experimental, and there is no data on which surfaces are dominant at a location.
        NOTE: our final objective is to make sure that environmental infection is around 10-15% of all transmissions to comply with the data.
        We do that via ENVIRONMENTAL_INFECTION_KNOB.

        Args:
            human (covid19sim.human.Human): `human` for whom to check environmental infection.

        Returns:
            (bool): whether `human` was infected via environmental contamination.
        """
        p_contamination = self.contamination_probability
        if not (human.is_susceptible and p_contamination > 0):
            return

        #
        p_transmission = get_environment_human_p_transmission(
                                        p_contamination,
                                        human,
                                        self.conf.get("ENVIRONMENTAL_INFECTION_KNOB"),
                                        self.conf['MASK_EFFICACY_FACTOR'],
                                        self.conf['HYGIENE_EFFICACY_FACTOR'],
                                        )

        x_environment =  self.rng.random() < p_transmission

        # track infection related stats
        human.city.tracker.track_infection(source="environment",
                                    from_human=None,
                                    to_human=human,
                                    location=self,
                                    timestamp=self.env.timestamp,
                                    p_infection=p_transmission,
                                    success=x_environment
                                )
        #
        if x_environment:
            human.infection_timestamp = human.env.timestamp
            compute_covid_properties(human)
            Event.log_exposed(self.conf.get('COLLECT_LOGS'), human, self, p_transmission, self.env.timestamp)

        return

    def is_open(self, date):
        """
        Checks if `self` is open on `date`.

        Args:
            date (datetime.date): date for which this is to be checked.

        Returns:
            (bool): True if its open
        """
        return date.weekday() in self.open_days

    def __hash__(self):
        """
        Hash of the location is the hash of its name

        Returns:
            int: hash
        """
        return hash(self.name)

    def serialize(self):
        """
        This function serializes the location object by deleting
        non-serializable keys

        Returns:
            dict: serialized location
        """
        s = self.__dict__
        if s.get('env'):
            del s['env']
        if s.get('rng'):
            del s['rng']
        if s.get('_env'):
            del s['_env']
        if s.get('contamination_timestamp'):
            del s['contamination_timestamp']
        if s.get('residents'):
            del s['residents']
        if s.get('humans'):
            del s['humans']
        return s


class Household(Location):
    """
    Household location class, inheriting from covid19sim.base.Location
    """
    def __init__(self, **kwargs):
        """
        Args:
            kwargs (dict): all the args necessary for a Location's init
        """
        super(Household, self).__init__(**kwargs)
        self.residents = []
        self.allocation_type = None

        location_type = kwargs['location_type']
        # for seniors, social common room serves as a workplace where they spend time hanging out with others
        if location_type == "SENIOR_RESIDENCE":
            self.n_nurses = 0
            name = kwargs.get("name")
            self.social_common_room = Location(
                env=kwargs.get("env"),
                rng=kwargs.get("rng"),
                name=f"SCR: {name}",
                conf=kwargs.get('conf'),
                location_type=location_type,
                lat=kwargs.get("lat"),
                lon=kwargs.get("lon"),
                area=kwargs.get("area"),
                capacity=None
            )

class School(Location):
    """
    School location class, inheriting from covid19sim.base.Location
    """
    def __init__(self, **kwargs):
        """
        Args:
            kwargs (dict): all the args necessary for a Location's init
        """
        super(School, self).__init__(**kwargs)
        self.n_students = 0
        self.n_teachers = 0

    def __repr__(self):
        return self.name + f"| Students:{self.n_students} Teachers:{self.n_teachers}"


class WorkplaceA(Location):
    """
    Stores location class, inheriting from covid19sim.base.Location
    """
    def __init__(self, **kwargs):
        """
        Args:
            kwargs (dict): all the args necessary for a Location's init
        """
        super(WorkplaceA, self).__init__(**kwargs)
        self.workers = set()
        self.n_workers = 0

    def assign_worker(self, human):
        """
        Adds `human` to the set of workers.

        Args:
            human (covi19sim.human.Human): `human` to add to the set of workers
        """
        self.workers.add(human)
        self.n_workers += 1

    def __repr__(self):
        return self.name + f"| {self.n_workers} workers"


class WorkplaceB(Location):
    """
    Stores location class, inheriting from covid19sim.base.Location
    """
    def __init__(self, **kwargs):
        """
        Args:
            kwargs (dict): all the args necessary for a Location's init
        """
        super(WorkplaceB, self).__init__(**kwargs)
        self.workers = set()
        self.n_workers = 0

    def assign_worker(self, human):
        """
        Adds `human` to the set of workers.

        Args:
            human (covi19sim.human.Human): `human` to add to the set of workers
        """
        self.workers.add(human)
        self.n_workers += 1

    def __repr__(self):
        return self.name + f"| {self.n_workers} workers"
