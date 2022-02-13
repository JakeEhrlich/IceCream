import random
import math
import numpy as np
from scipy.stats import truncnorm

# Entity is a glorified dictionary that
# restricts its keys to types and is
# hashable by identity.
class Entity:

    def __init__(self):
        self.components = {}

    def add_component(self, component_type, *args, **kwargs):
        assert type(component_type) == type
        self.components[component_type] = component_type(*args, **kwargs)

    def has_component(self, component_type):
        assert type(component_type) == type
        return component_type in self.components

    def get_component(self, component_type):
      assert type(component_type) == type
      if component_type in self.components:
          return self.components[component_type]
      return None

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return hash(id(self))

    def __getitem__(self, key):
        return self.get_component(key)

    def __setitem__(self, key, value):
        self.add_component[key] = value

    def __contains__(self, key):
        return self.has_component(key)

cryoscopic_constant_of_water = 1853.0 # given in K·g/mol

# Right now all soluable components are assumed
# fully disolved and anything that is not
# a soluable component is assumed not-disolved.
# Later we can try to account for how much of
# each good is disolved and how much parcipitates
# out. This is useful for calculating freezing point
# depression.
class SolubleComponent:
    def __init__(self, atomic_mass, van_t_hoff_factor=1.0):
        self.atomic_mass = atomic_mass # given in g/mol
        self.van_t_hoff_factor = van_t_hoff_factor # unitless ratio

# This just tells you if an entity is Water or not.
# "solids" are defined as "thigns that are not water".
# So this means that fats, and alcohol are conisderd solids
# along with other disolved materials like disolved sugars.
class WaterComponent:
    def __init__(self):
        pass

# It turns out that modeling SFC is *super* hard
# so we're not going to do it for now. But we
# do still want to account for total fat. So this
# acts like Water and Bulk for now.
class FatComponent:
    def __init__(self):
        pass

# This is used for measuring brix
class SugarComponent:
    def __init__(self, relative_sweetness):
        self.relative_sweetness = relative_sweetness

# This is used for measuring bulk ingrdients that have
# no effect other than adding bulk, they are solids that
# display water and thats about it.
class BulkComponent:
    def __init__(self):
        pass

class AshComponent:
    def __init__(self):
        pass

class SaltyComponent:
    def __init__(self, saltiness_index):
        # Most Ash components have a saltiness of 0.0
        self.saltiness_index = saltiness_index # This is given in mass/mass, not mol/mol

class Mixture:
    def __init__(self, mix = None):
        self.components_ = {}

    def normalize(self):
        total_index = sum(self.components_.values())
        for e in self.components_:
            self.components_[e] /= total_index

    def add_entity(self, entity, index):
        # component is some non-mixture component
        assert type(entity) == Entity
        assert type(index) == float
        assert index > 0.0
        assert index <= 1.0
        if entity in self.components_:
            self.components_[entity] += index
        else:    
            self.components_[entity] = index

    def entity_index(self, entity):
        if entity in self.components_:
            return self.components_[entity]
        return 0.0

    def add_mixture(self, mixture, index):
        assert type(mixture) == Mixture
        for (e, v) in mixture.components_.items():
          if e in self.components_:
              self.components_[e] += index * v
          else:
              self.components_[e] = index * v

    def solids_index(self):
        return 1.0 - self.water_index()

    def index(self, component):
        return sum([index for (e, index) in self.components_.items() if component in e])

    def water_index(self):
        return self.index(WaterComponent)

    def sum_fn_over(self, componet, fn):
        return sum([fn(index, s) for (e, index) in self.components_.items() if (s := e[componet])])

    def ice_index(self, temperature):
        # We pretend as though our mixture is 1g in mass. The final result is grams water - grams_unfrozen_water
        # which might as well by the ice index.

        # Each component depresses the freezing point by m*atomic_mass*van_t_hoff/total_unfrozen_water
        # We factor out the cryoscopic constant and total_unfrozen_water.
        total_depression_g_per_cryo = self.sum_fn_over(SolubleComponent, lambda index, s: index / s.atomic_mass * s.van_t_hoff_factor)
            
        # Now freeze_point_depression_kg gives freezing point depression when devided by total_unfrozen_water
        freeze_point_depression_g = cryoscopic_constant_of_water * total_depression_g_per_cryo
        # Technically this is the drop in freezing point in Kelvin...but thats also the drop in Celcius...and the freezing
        # point of celcius is 0F...so uh we just negate it lol
        init_freeze_point = -freeze_point_depression_g / self.water_index()
        if temperature >= init_freeze_point:
            return 0.0
        # Now since we're know we have some partial freezing we know that
        # temperature = freezing_point_depression = freeze_point_depression_kg / total_unfrozen_watter
        # thus freeze_point_depression_kg / temperature = total_unfrozen_watter
        total_unfrozen_water = -freeze_point_depression_g / temperature # This formula *demands* celcius not Kelvin
        # After that calculating the ice content is just the difference
        return (self.water_index() - total_unfrozen_water) / self.water_index()

    def fat_index(self):
        return self.index(FatComponent)

    def brix(self):
        total_sweetness = self.sum_fn_over(SugarComponent, lambda index, s: index * s.relative_sweetness)
        return 100.0 * total_sweetness

    def sugars_index(self):
        return self.index(SugarComponent)

    def saltiness_index(self):
        total_saltiness = self.sum_fn_over(SaltyComponent, lambda index, s: index * s.saltiness_index)
        return total_saltiness

    def ash_index(self):
        return self.index(AshComponent)

    def non_fat_solids_index(self):
        return self.solids_index() - self.fat_index()

class Formula:
    def __init__(self, formula = None):
        if formula is None:
            self.mixtures_ = {}
        else:
            self.mixtures_ = dict(formula.mixtures_)

    def add_mixture(self, name, mixture, index):
        self.mixtures_[name] = (mixture, index)
        
    def add_entity(self, name, entity, index):
        mono_mix = Mixture()
        mono_mix.add_entity(entity, 1.0)
        self.mixtures_[name] = (mono_mix, index)

    def mixture(self):
        out = Mixture()
        for (_, (mix, index)) in self.mixtures_.items():
            out.add_mixture(mix, index)
        out.normalize() # Just in case?
        return out

    def formula(self):
        return {name: index for (name, (_, index)) in self.mixtures_.items()}

    # Before optimization, you must be normalized
    def normalize(self):
        total_index = sum([index for (_, index) in self.mixtures_.values()])
        for (e, (m, index)) in self.mixtures_.items():
            self.mixtures_[e] = (m, index / total_index)

    def mutate(self, scale=1.0):
        (name, (mix, index)) = random.choice(list(self.mixtures_.items()))
        diff = np.random.normal(0.0, scale)
        self.mixtures_[name] = (mix, max(index + diff, 0.00001))
        self.normalize()

def clamp(num, min_value, max_value):
   return max(min(num, max_value), min_value)

def simulated_annealing(formula, schedule, cost, max_steps):
    prev_cost = cost(formula)
    min_formula = formula
    min_cost = prev_cost
    for step in range(max_steps):
        temp = schedule(1 - (step + 1.0) / max_steps, step)
        new_formula = Formula(formula)
        # Scaling the variance of the mutation was found to be crucial to convergence
        new_formula.mutate(min(1.0, temp))
        new_cost = cost(new_formula)
        # Clamping the expoential into a more sensible range was found to be critical to sucess
        # because costs can vary greatly from one iteration to the next.
        if math.exp(clamp(-(new_cost - prev_cost) / temp, -100, 100)) >= random.uniform(0.0, 1.0):
            if new_cost < min_cost:
                min_formula = new_formula
                min_cost = new_cost
            prev_cost = new_cost
            formula = new_formula
    return min_formula

def linear_schedule(init_temp):
    return lambda x, step: init_temp * x

def quadratic_schedule(init_temp):
    return lambda x, step: init_temp * x * x

# Use alpha between 0.8 and 0.9
def exponetial_schedule(init_temp, alpha=0.85):
    return lambda x, step: init_temp * (alpha ** step)

# Use alpha > 1
def log_schedule(init_temp, alpha=1.5):
    return lambda x, step: init_temp / (1 + alpha * math.log(step + 1))

# The linear step schedule was found to work quite well oddly,
# I would have expected the log step to work better but perhaps it does not
# cool quickly enough to optimize in a truely fine tuned way.
# Use alpha > 0
def linear_step_schedule(init_temp, alpha=1.0):
    return lambda x, step: init_temp / (1 + alpha * step)

def quadratic_step_schedule(init_temp, alpha=1.0):
    return lambda x, step: init_temp / (1 + alpha * step * step)

# TODO: It would be really great if we could read these in from json into
#       a dictionary.

## Sweetners
# TODO: Add artifical sweetners
Allulose = Entity()
Allulose.add_component(SolubleComponent, atomic_mass=180.156)
Allulose.add_component(SugarComponent, relative_sweetness=0.7)

Glucose = Entity()
Glucose.add_component(SolubleComponent, atomic_mass=180.156)
Glucose.add_component(SugarComponent, relative_sweetness=0.7)

Sucrose = Entity()
Sucrose.add_component(SolubleComponent, atomic_mass=342.3)
Sucrose.add_component(SugarComponent, relative_sweetness=1.0)

GlucoseDE42 = Entity()
GlucoseDE42.add_component(SolubleComponent, atomic_mass=428.0) # Can I get this more accurate?
GlucoseDE42.add_component(SugarComponent, relative_sweetness=0.3)

Fructose = Entity()
Fructose.add_component(SolubleComponent, atomic_mass=180.156)
Fructose.add_component(SugarComponent, relative_sweetness=1.7)

Lactose = Entity()
Lactose.add_component(SolubleComponent, atomic_mass=180.156)
Lactose.add_component(SugarComponent, relative_sweetness=0.3)

## Alcohols
# TODO: Add Manitol and Isomalt
Sorbitol = Entity()
Sorbitol.add_component(SolubleComponent, atomic_mass=182.17)
Sorbitol.add_component(SugarComponent, relative_sweetness=0.5)

Xylitol = Entity()
Xylitol.add_component(SolubleComponent, atomic_mass=182.17)
Xylitol.add_component(SugarComponent, relative_sweetness=1.0)

Ethonol = Entity()
Ethonol.add_component(SolubleComponent, atomic_mass=46.07, van_t_hoff_factor=1.0)

## Water
Water = Entity()
Water.add_component(WaterComponent)

## Fats
ButterOil = Entity()
ButterOil.add_component(FatComponent)

CocoaButter = Entity()
CocoaButter.add_component(FatComponent)

RefinedCoconutFat = Entity()
RefinedCoconutFat.add_component(FatComponent)

VegtableOil = Entity()
VegtableOil.add_component(FatComponent)

## Salts
SodiumChloride = Entity()
SodiumChloride.add_component(SolubleComponent, atomic_mass=58.44, van_t_hoff_factor=2.0)
SodiumChloride.add_component(SaltyComponent, saltiness_index=1.0)
SodiumChloride.add_component(AshComponent)

Potassium = Entity()
Potassium.add_component(SolubleComponent, atomic_mass=39.0983, van_t_hoff_factor=1.0)
Potassium.add_component(SaltyComponent, saltiness_index=0.6)
Potassium.add_component(AshComponent)

PotassiumChloride = Entity()
PotassiumChloride.add_component(SolubleComponent, atomic_mass=74.5513, van_t_hoff_factor=2.0)
PotassiumChloride.add_component(SaltyComponent, saltiness_index=0.6)
PotassiumChloride.add_component(AshComponent)

# I can add Calcium and other stuff later perhaps?
# A free-floating Calcium ion would be considered soluble with Van T Hoff factor of 1.0 I think.
GenericAsh = Entity()
GenericAsh.add_component(AshComponent)

## Generics

GenericBulk = Entity()
GenericBulk.add_component(BulkComponent)

GenericStabEmul = Entity()
GenericStabEmul.add_component(BulkComponent)

GenericProtien = Entity()
GenericProtien.add_component(BulkComponent)

## Stabilizers and Emulsifiers

Polysorbate80 = Entity()
Polysorbate80.add_component(BulkComponent)

SoyLecithin = Entity()
SoyLecithin.add_component(BulkComponent)

SoyLecithinPowder = Entity()
SoyLecithinPowder.add_component(BulkComponent)

EggYolkPowder = Entity()
EggYolkPowder.add_component(BulkComponent)

SodiumCasinate = Entity()
SodiumCasinate.add_component(BulkComponent)

MonoDiglycerideFlakes = Entity()
MonoDiglycerideFlakes.add_component(BulkComponent)

SodiumAlginate = Entity()
SodiumAlginate.add_component(BulkComponent)

CarboxymethylCellulose = Entity()
CarboxymethylCellulose.add_component(BulkComponent)
CMC = CarboxymethylCellulose

IotaCarrageenan = Entity()
IotaCarrageenan.add_component(BulkComponent)

KappaCarrageenan = Entity()
KappaCarrageenan.add_component(BulkComponent)

LambdaCarrageenan = Entity()
LambdaCarrageenan.add_component(BulkComponent)

SucroseEsters = Entity()
SucroseEsters.add_component(BulkComponent)

Stab210s = Entity()
Stab210s.add_component(BulkComponent)

LocustBeanGum = Entity()
LocustBeanGum.add_component(BulkComponent)

GuarGum = Entity()
GuarGum.add_component(BulkComponent)

XanthanGum = Entity()
XanthanGum.add_component(BulkComponent)

CelluloseGum = Entity()
CelluloseGum.add_component(BulkComponent)

MPPerfectGelato = Entity()
MPPerfectGelato.add_component(BulkComponent)

MPPerfectSorbet = Entity()
MPPerfectSorbet.add_component(BulkComponent)

MPPerfectIceCream = Entity()
MPPerfectIceCream.add_component(BulkComponent)

## Dairy Products
NFMS = Mixture()
NFMS.add_entity(Lactose, 0.05)
NFMS.add_entity(GenericAsh, 0.0075)
NFMS.add_entity(GenericProtien, 0.0325)
NFMS.normalize()

SkimMilkPowder = Mixture()
SkimMilkPowder.add_mixture(NFMS, 0.98)
SkimMilkPowder.add_entity(Water, 0.02)
SkimMilkPowder.normalize()

SkimMilk = Mixture()
SkimMilk.add_mixture(NFMS, 0.09)
SkimMilk.add_entity(Water, 0.87)
SkimMilk.normalize()

Cream = Mixture()
Cream.add_entity(ButterOil, 0.35)
Cream.add_mixture(SkimMilk, 0.65)

WholeMilk = Mixture()
WholeMilk.add_entity(ButterOil, 0.03)
WholeMilk.add_mixture(SkimMilk, 0.97)

JerseyMilk = Mixture()
JerseyMilk.add_entity(ButterOil, 0.05)
JerseyMilk.add_mixture(SkimMilk, 0.95)

# Extras
CocoaPowder = Mixture()
CocoaPowder.add_entity(CocoaButter, 0.21)
CocoaPowder.add_entity(GenericProtien, 0.14)
CocoaPowder.add_entity(Potassium, 0.045)
CocoaPowder.add_entity(GenericBulk, 1.0 - 0.21 - 0.14 - 0.045)

VanillinPowder = Entity()
VanillinPowder.add_component(BulkComponent)

Aperol = Mixture()
Aperol.add_entity(Sucrose, 0.35)
Aperol.add_entity(Ethonol, 0.11)
Aperol.add_entity(Water, 1.0 - 0.35 - 0.11)

def strawberry_juice(observed_brix):
    # Sourced from foodb.ca
    expected_water_index = 0.78100
    expected_sucrose_index = 0.12685
    expected_fructose_index = 0.00930
    expected_glucose_index = 0.008175

    return produce_juice(observed_brix, expected_water_index, expected_sucrose_index, expected_fructose_index, expected_glucose_index)

def carrot_juice(observed_brix):
    # Sourced from foodb.ca
    # TODO: Carrot has non-trivial fat in it, we need to add that in.
    expected_water_index = 0.74625
    expected_sucrose_index = 0.03083718
    expected_fructose_index = 0.01140172
    expected_glucose_index = 0.08 + 0.01536

    return produce_juice(observed_brix, expected_water_index, expected_sucrose_index, expected_fructose_index, expected_glucose_index)

def produce_juice(observed_brix, expected_water_index, expected_sucrose_index, expected_fructose_index, expected_glucose_index):
    # We assume that the ratio of water:solids and the ratio of sucrose:glucose:fructose is the same for all
    # strawberry juice (its not), Additionally we assume 1g of any sugar will effect the index of refraction the same way
    # which is aproximently true but is an aproximation. With those assumptions we can improve our mixture accuracy based
    # only on the observed brix from a refractometer. Lastly we assume the juicer removes 85% of other bulk. This is a 
    # very very rough aproximation but thats the nature of making sorbet like this.

    # Need to know the ratio of water and solids
    expected_other_solids = 1.0 - expected_water_index - expected_sucrose_index - expected_fructose_index - expected_glucose_index

    # Calculate total sugars and non-sugars based on observation
    assumed_total_sugars = observed_brix / 100.0
    assumed_total_non_sugars = 1.0 - assumed_total_sugars

    # Calculate totals for ratios
    expected_total_non_sugars = expected_water_index + expected_other_solids
    expected_total_sugars = expected_sucrose_index + expected_fructose_index + expected_glucose_index

    # Calculate total sugars wtih sugar ratios adjusted for observed brix
    assumed_sucrose = assumed_total_sugars * expected_sucrose_index / expected_total_sugars
    assumed_fructose = assumed_total_sugars * expected_fructose_index / expected_total_sugars
    assumed_glucose = assumed_total_sugars * expected_glucose_index / expected_total_sugars

    # Calculate water+solids adjusted by observation
    assumed_water = assumed_total_non_sugars * expected_water_index / expected_total_non_sugars
    assumed_bulk = 0.15 * assumed_total_non_sugars * expected_other_solids / expected_total_non_sugars

    # Construct mixture
    Juice = Mixture()
    Juice.add_entity(Sucrose, assumed_sucrose)
    Juice.add_entity(Fructose, assumed_fructose)
    Juice.add_entity(Glucose, assumed_glucose)
    Juice.add_entity(GenericBulk, assumed_bulk)
    Juice.add_entity(Water, assumed_water)
    Juice.normalize() # Because we left out some bulk we normalize

    return Juice

# Tweak the base_formula to add/remove ingredients
# TODO: Consider adding an abstract "cost" param
base_formula = Formula()
#base_formula.add_mixture("WholeMilk", WholeMilk, 0.2)
#base_formula.add_mixture("Cream", Cream, 0.2)
#base_formula.add_mixture("SkimMilkPowder", SkimMilkPowder, 0.2)
base_formula.add_mixture("CocoaPowder", CocoaPowder, 0.2)
#base_formula.add_entity("GlucoseDE42", GlucoseDE42, 0.2)
#base_formula.add_entity("Fructose", Fructose, 0.2)
base_formula.add_entity("Glucose", Glucose, 0.2)
base_formula.add_entity("Sucrose", Sucrose, 0.2)
#base_formula.add_entity("CocoaButter", CocoaButter, 0.2)
#base_formula.add_entity("RefinedCoconutFat", RefinedCoconutFat, 0.2)
#base_formula.add_entity("StabEmul", GenericStabEmul, 0.2)
base_formula.add_entity("MP Perfect Gelato", MPPerfectGelato, 0.2)
#base_formula.add_entity("MP Perfect Ice Cream", GenericStabEmul, 0.2)
#base_formula.add_entity("MP Perfect Sorbet", GenericStabEmul, 0.2)
base_formula.add_entity("Salt", SodiumChloride, 0.2)
#base_formula.add_entity("Ethonol", Ethonol, 0.2)
base_formula.add_entity("Water", Water, 0.2)
base_formula.normalize()

# Helper to print a recipe in a specified weight
def print_recipe(formula, grams):
    for (k, v) in formula.items():
        print("%s:" % k, grams * v)

def scale(x, vMin, vMax):
    return (x - vMin) / (vMax - vMin)

# Helper for ensuring even small values are weighted correctly so nothing gets left
# by the way-side (you have to scale small values more to get them correct otherwise)
def scaled_least_squares(tuples):
    cost = 0.0
    for (target, vMin, vMax, v) in tuples:
        # Scale everything to [0, 1]
        v = scale(v, vMin, vMax)
        target = scale(target, vMin, vMax)
        # Further more we also divide by the target so that small values have extra advantage.
        # This seems to bias things a bit *too* far however
        cost += ((v - target) / target)**2.0

    return math.sqrt(cost)

gelato_base_formula = Formula()
gelato_base_formula.add_mixture("WholeMilk", WholeMilk, 0.2)
gelato_base_formula.add_mixture("Cream", Cream, 0.2)
gelato_base_formula.add_mixture("SkimMilkPowder", SkimMilkPowder, 0.2)
gelato_base_formula.add_entity("Sucrose", Sucrose, 0.2)
gelato_base_formula.add_entity("MP Perfect Gelato", MPPerfectGelato, 0.2)
gelato_base_formula.add_entity("Salt", SodiumChloride, 0.2)
gelato_base_formula.add_entity("Water", Water, 0.2)
gelato_base_formula.normalize()

def gelato_base_cost_fn(formula):
    mix = formula.mixture()
    formula = formula.formula()

    cost = scaled_least_squares([
        # %fat
        (6.0,  0.0, 100.0, 100*mix.fat_index()),
        # sweetness (in brix)
        (14.0, 0.0, 100.0, mix.brix()),
        # %stab/emul
        (0.15,  0.0, 100.0, 100*mix.entity_index(MPPerfectGelato)),
        # %water
        (60.0, 0.0, 100.0, 100*mix.water_index()),
        # %ice at -6.0C
        (50.0, 0.0, 100.0, 100*mix.ice_index(-6.0)),
        # saltiness (in brix-like units but relative to sodium chloride instead of sucrose)
        (0.1,  0.0, 100.0, 100*mix.saltiness_index())
    ])

    return cost

pure_choc_formula = Formula()
pure_choc_formula.add_mixture("CocoaPowder", CocoaPowder, 0.2)
pure_choc_formula.add_entity("Glucose", Glucose, 0.2)
pure_choc_formula.add_entity("Sucrose", Sucrose, 0.2)
pure_choc_formula.add_entity("MP Perfect Gelato", MPPerfectGelato, 0.2)
pure_choc_formula.add_entity("Salt", SodiumChloride, 0.2)
pure_choc_formula.add_entity("Water", Water, 0.2)
pure_choc_formula.normalize()

def pure_chocolate_cost_fn(formula):
    mix = formula.mixture()
    formula = formula.formula()

    cost = scaled_least_squares([
        # %fat
        (4.0,  0.0, 100.0, 100*mix.fat_index()),
        # sweetness (in brix)
        (14.0, 0.0, 100.0, mix.brix()),
        # %stab/emul
        (0.15,  0.0, 100.0, 100*mix.entity_index(MPPerfectGelato)),
        # %water
        (60.0, 0.0, 100.0, 100*mix.water_index()),
        # %ice at -6.0C
        (50.0, 0.0, 100.0, 100*mix.ice_index(-6.0)),
        # saltiness (in brix-like units but relative to sodium chloride instead of sucrose)
        (0.5,  0.0, 100.0, 100*mix.saltiness_index()),
        # Amt of Cocoa Powder as a %percent
        #(11.0, 0.0, 100.0, 100*mix.entity_index(CocoaPowder))
    ])

    return cost

# NOTE: You need to add some Vanillin powder for full effect, I've read you can use 1/4th by *volume*
#       what you would use in extract. 5 grams of extract per 1000g mix is a nice subtle flavor.
#       4oz appears to take up a bit over a cup so this stuff has a bit less than half the density
#       of water (but also extract is a bit lighter than water). So I *think* about 1/8th of 5 grams
#       would be a good start, lets call it 0.5g in 1000g which is uh...0.05%?       
marshmallow_formula = Formula()
marshmallow_formula.add_entity("GlucoseDE42", GlucoseDE42, 0.2)
marshmallow_formula.add_entity("VanillinPowder", VanillinPowder, 0.2)
marshmallow_formula.add_entity("RefinedCoconutFat", RefinedCoconutFat, 0.2)
marshmallow_formula.add_entity("MP Perfect Ice Cream", MPPerfectIceCream, 0.2)
marshmallow_formula.add_entity("Polysorbate 80", Polysorbate80, 0.2)
marshmallow_formula.add_entity("Salt", SodiumChloride, 0.2)
marshmallow_formula.add_entity("Water", Water, 0.2)
marshmallow_formula.normalize()
def marshmallow_cost_fn(formula):
    mix = formula.mixture()
    formula = formula.formula()

    cost = scaled_least_squares([
        # %fat
        (6.0,  0.0, 100.0, 100*mix.fat_index()),
        # sweetness (in brix)
        (10.0, 0.0, 100.0, mix.brix()),
        # %stab/emul
        (0.2,  0.0, 100.0, 100*mix.entity_index(MPPerfectIceCream)),
        # %stab/emul
        (0.3,  0.0, 100.0, 100*mix.entity_index(Polysorbate80)),
        # Vanillin
        (0.05,  0.0, 100.0, 100*mix.entity_index(VanillinPowder)),
        # %water
        (60.0, 0.0, 100.0, 100*mix.water_index()),
        # %ice at -6.0C
        (50.0, 0.0, 100.0, 100*mix.ice_index(-6.0)),
        # saltiness (in brix-like units but relative to sodium chloride instead of sucrose)
        (0.3,  0.0, 100.0, 100*mix.saltiness_index()),
    ])

    return cost

strawberry_sorbet = Formula()
strawberry_sorbet.add_mixture("Juice", strawberry_juice(12.0), 0.2)
strawberry_sorbet.add_entity("MP Perfect Sorbet", MPPerfectSorbet, 0.2)
#strawberry_sorbet.add_entity("Salt", SodiumChloride, 0.2)
strawberry_sorbet.add_mixture("Aperol", Aperol, 0.2)
strawberry_sorbet.normalize()

def strawberry_sorbet_cost_fn(formula):
    mix = formula.mixture()
    formula = formula.formula()

    cost = scaled_least_squares([
        # sweetness (in brix)
        #(16.0, 0.0, 100.0, mix.brix()),
        # %stab/emul
        (0.5,  0.0, 100.0, 100*mix.entity_index(MPPerfectSorbet)),
        # %ice at -6.0C
        (50.0, 0.0, 100.0, 100*mix.ice_index(-6.0)),
        # saltiness (in brix-like units but relative to sodium chloride instead of sucrose)
        #(0.3,  0.0, 100.0, 100*mix.saltiness_index()),
    ])

    return cost


def dump_recipe(base_formula, cost_fn, recipe_name, total_weight):
    print("init cost recipe_name: ", cost_fn(base_formula))
    base_formula = simulated_annealing(base_formula, linear_step_schedule(init_temp=200.0), cost_fn, max_steps=10000)
    print("final cost recipe_name: ", cost_fn(base_formula))

    mix = base_formula.mixture()
    print("\nFacts:")
    print("fat: ", mix.fat_index())
    print("brix: ", mix.brix())
    print("stab/emul: ", mix.entity_index(MPPerfectIceCream))
    print("water: ", mix.water_index())
    print("ice at -6C: ", mix.ice_index(-6.0))
    print("saltiness: ", mix.saltiness_index())
    print("solids: ", mix.solids_index())

    print("\nRecipe %s (total = %fg):" % (recipe_name, total_weight))
    formula = base_formula.formula()
    print_recipe(formula, total_weight)

dump_recipe(strawberry_sorbet, strawberry_sorbet_cost_fn, "strawberry_sorbet", 850)