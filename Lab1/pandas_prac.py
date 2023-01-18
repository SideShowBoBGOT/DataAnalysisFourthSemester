import pandas as pd
import numpy as np

df = pd.DataFrame([['A'] for _ in range(5)])
df = pd.concat([df, pd.DataFrame([['B'] for _ in range(5)], index=[5, 6, 7, 8, 9])])
df = pd.concat(
    [
        df.reset_index(drop=True),
        pd.DataFrame(
            [[np.random.rand() for _ in range(10)] for _ in range(10)],
            columns=[x for x in range(1, 11)]
        ).reset_index(drop=True)
    ],
    axis=1
)

bosses = pd.DataFrame(
    {
        'id': [133, 111, 876, 987],
        'name': ['Mona Sax', 'Tony Soprano', 'John Smith', 'Mark Webb']
    }
)

employees = pd.DataFrame(
    {
        'id': [123, 432, 911, 678, 422],
        'name': ['Max Dylon', 'Kurt Russel', 'Linda Hamilton', 'Ruby Rails', 'Corey Star'],
        'boss_id': [111, 133, 876, 987, 111]
    }
)

boss_empl = pd.merge(bosses, employees, left_on='id', right_on='boss_id').iloc[:, :4]
boss_empl.columns = ['boss_id', 'boss_name', 'employee_id', 'employees_name']

df = pd.DataFrame(['A'] for _ in range(5))
df = pd.concat([df, pd.DataFrame(['B'])]).reset_index(drop=True)
s = pd.DataFrame([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]).transpose()
s.columns = ['a', 'b']
#print(df.iloc[:, 0].isin(['A']))
#print(s['a'][0].__class__)