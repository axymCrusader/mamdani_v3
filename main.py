import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class FuzzyVariable:
    def __init__(self, universe, name):
        self.universe = universe
        self.name = name
        self.terms = {}

    def __getitem__(self, term):
        return self.terms.get(term)

    def __setitem__(self, term, mf):
        self.terms[term] = mf


class FuzzyInferenceSystem:
    def __init__(self, selected_func, num_intervals, data_from_csv, intervals_data):
        self.selected_func = selected_func
        self.num_intervals = num_intervals
        self.data_from_csv = data_from_csv
        self.intervals_data = intervals_data
        self.variables = self.create_fuzzy_variables()

    def create_fuzzy_variables(self):
        variables = {
            'battery_power': FuzzyVariable(
                np.linspace(self.data_from_csv["battery_power"].min(), self.data_from_csv["battery_power"].max(), 100),
                'battery_power'),
            'ram': FuzzyVariable(np.linspace(self.data_from_csv["ram"].min(), self.data_from_csv["ram"].max(), 100),
                                 'ram'),
            'px': FuzzyVariable(np.linspace(self.data_from_csv["px"].min(), self.data_from_csv["px"].max(), 100), 'px'),
            'price_range': FuzzyVariable(np.linspace(0, 3, 100), 'price_range')
        }

        self.generate_membership_functions(variables)
        return variables

    def generate_membership_functions(self, variables):
        for var_name, var in variables.items():
            if var_name in self.intervals_data:
                terms = self.intervals_data[var_name]
                intervals = np.linspace(var.universe.min(), var.universe.max(), self.num_intervals + 1)

                for term, (start, end) in zip(terms, zip(intervals, intervals[1:])):
                    middle = (start + end) / 2
                    if self.selected_func == 'trimf':
                        var.terms[term] = [start, middle, end]
                    elif self.selected_func == 'trapmf':
                        q1 = start + (end - start) / 4
                        q3 = end - (end - start) / 4
                        var.terms[term] = [start, q1, q3, end]
                    elif self.selected_func == 'gaussmf':
                        var.terms[term] = [middle, (end - start) / 4]

    def plot_membership_functions(self, var):
        plt.figure(figsize=(10, 5))
        for term, params in var.terms.items():
            if self.selected_func == 'trimf':
                plt.plot(var.universe, self.trimf(var.universe, params), label=term)
            elif self.selected_func == 'trapmf':
                plt.plot(var.universe, self.trapmf(var.universe, params), label=term)
            elif self.selected_func == 'gaussmf':
                plt.plot(var.universe, self.gaussmf(var.universe, params), label=term)
        plt.title(f'Membership Functions for {var.name}')
        plt.xlabel(var.name)
        plt.ylabel('Membership Degree')
        plt.legend()
        plt.grid()
        plt.show()

    @staticmethod
    def trimf(x, abc):
        a, b, c = abc
        y = np.maximum(0, np.minimum((x - a) / (b - a), (c - x) / (c - b)))
        return y

    @staticmethod
    def trapmf(x, abcd):
        a, b, c, d = abcd
        y = np.maximum(0, np.minimum((x - a) / (b - a), np.minimum(1, (d - x) / (d - c))))
        return y

    @staticmethod
    def gaussmf(x, mean_sigma):
        mean, sigma = mean_sigma
        return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

    def infer_for_row(self, row):
        bp, r, p = row['battery_power'], row['ram'], row['px']

        bp_memberships = {term: self.trimf(self.variables['battery_power'].universe, params) for term, params in
                          self.variables['battery_power'].terms.items()}
        ram_memberships = {term: self.trimf(self.variables['ram'].universe, params) for term, params in
                           self.variables['ram'].terms.items()}
        px_memberships = {term: self.trimf(self.variables['px'].universe, params) for term, params in
                          self.variables['px'].terms.items()}

        rule_outputs = []
        for bp_term in self.intervals_data['battery_power']:
            for ram_term in self.intervals_data['ram']:
                for px_term in self.intervals_data['px']:
                    bp_idx = np.abs(self.variables['battery_power'].universe - bp).argmin()
                    ram_idx = np.abs(self.variables['ram'].universe - r).argmin()
                    px_idx = np.abs(self.variables['px'].universe - p).argmin()

                    rule_strength = min(
                        bp_memberships[bp_term][bp_idx],
                        ram_memberships[ram_term][ram_idx],
                        px_memberships[px_term][px_idx]
                    )

                    if rule_strength > 0:
                        for price_term in self.intervals_data['price_range']:
                            rule_outputs.append((rule_strength, price_term))

        if rule_outputs:
            numerator = sum(strength * np.sum(
                self.variables['price_range'].universe * self.trimf(self.variables['price_range'].universe,
                                                                    self.variables['price_range'].terms[term])) for
                            strength, term in rule_outputs)
            denominator = sum(strength * np.sum(
                self.trimf(self.variables['price_range'].universe, self.variables['price_range'].terms[term])) for
                              strength, term in rule_outputs)
            result = numerator / denominator if denominator != 0 else np.mean(self.variables['price_range'].universe)
        else:
            result = np.mean(self.variables['price_range'].universe)

        return result

    def train(self, training_data, learning_rate=0.01, epochs=100):
        for epoch in range(epochs):
            total_error = 0
            for _, row in training_data.iterrows():
                actual = row['price_range']
                predicted = self.infer_for_row(row)
                error = actual - predicted
                total_error += error ** 2

                for var in self.variables.values():
                    for term, params in var.terms.items():
                        if self.selected_func == 'trimf' and len(params) == 3:
                            params[1] += learning_rate * error
                        elif self.selected_func == 'trapmf' and len(params) == 4:
                            params[1] += learning_rate * error / 2
                            params[2] += learning_rate * error / 2
                        elif self.selected_func == 'gaussmf' and len(params) == 2:
                            params[0] += learning_rate * error

            print(f'Epoch {epoch + 1}/{epochs}, Total Error: {total_error}')

    def predict(self, test_data):
        return np.array([self.infer_for_row(row) for _, row in test_data.iterrows()])


if __name__ == '__main__':
    selected_func = 'trimf'
    num_intervals = 3

    train_data = pd.read_csv('train.csv')


    test_data = pd.read_csv('test.csv')

    intervals_data = {
        'battery_power': ['Little', 'Medium', 'Much'],
        'ram': ['Little', 'Medium', 'Many'],
        'px': ['Little', 'Medium', 'Many'],
        'price_range': ['Cheap', 'Optimal', 'Expensive']
    }

    fuzzy_system = FuzzyInferenceSystem(selected_func, num_intervals, train_data, intervals_data)

    fuzzy_system.train(train_data, learning_rate=0.01, epochs=100)

    predictions = fuzzy_system.predict(test_data)
    print(predictions)
    for i, row in test_data.iterrows():
        print(f"Данные: battery_power={row['battery_power']}, ram={row['ram']}, px={row['px']}")
        print(f"Предсказанный price_range: {predictions[i]:}")
        price_range_str = "Дешевый" if predictions[i] <= 1 else "Средний" if predictions[i] <= 2 else "Дорогой"
        print(f"Интерпретация: {price_range_str}")
        print("-" * 20)
