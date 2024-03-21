import csv
import numpy as np

PRECISION = 5

def read_csv_file(file_name):
    data = []
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return data

def write_csv_file(data, file_name):
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def remove_columns(data, columns_to_remove):
    header = data[0]
    indices_to_remove = [header.index(column) for column in columns_to_remove]
    for row in data:
        for index in sorted(indices_to_remove, reverse=True):
            del row[index]
            
def change_labels(data):
    idx = data[0].index("LoanStatus")
    data[0].pop(idx)
    data[0].append("y")
    for row in data[1:]:
        if row[idx] == "PIF":
            row.append(0)
        elif row[idx] == "CHGOFF":
            row.append(1)
        row.pop(idx)

def add_intercept(data):
    data[0].insert(0,"Intercept")
    for row in data[1:]:
        row.insert(0,1)

def norm_columns(data, cols):
    data_array = np.array(data[1:], dtype=float)
    for col in cols:
        idx = data[0].index(col)
        temp = data_array[:,idx] - np.mean(data_array[:,idx])
        if np.std(data_array[:,idx]) != 0:
            temp = temp / np.std(data_array[:,idx])
            data_array[:,idx] = np.round(temp, PRECISION)
    for i, row in enumerate(data[1:]):
        data[i+1] = data_array[i].tolist()

def normalize_all(data):
    columns_to_normalize = ['ThirdPartyDollars', 'GrossApproval', 'TermInMonths', 'Log S&P Open', "SP500_Past_Month_Return","SP500_Past_YR_Return", "US_Med_Home_Price_Month_Change", "US_Med_Home_Price_YR_Change", "Borr_State_Unemployment_YR_Change", "Proj_State_Unemployment_YR_Change", "Borr_State_GDP_YR_Change","Proj_State_GDP_YR_Change", "Borr_State_Income_YR_Change", "Proj_State_Income_YR_Change", "Borr_State_Vacancy_YR_Change", "Proj_State_Vacancy_YR_Change", "Borr_State_Homeowner_YR_Change", "Proj_State_Homeowner_YR_Change", "Fed_Fund_Month_Change", "Fed_Fund_YR_Change", "CCI_Month_Change", "BCI_Month_Change", "CLI_Month_Change", "CCI_YR_Change", "BCI_YR_Change", "CLI_YR_Change"]

    columns_to_normalize += ["Term Multiple", "Same State", "In CA", "Is ThirdParty", "Missing Interest", "Refinance", "Delta", "Private Sector", "Premier", "CORPORATION", "INDIVIDUAL", "Missing Business", "PARTNERSHIP", "Borr_State_Unemployment_Missing", "Proj_State_Unemployment_Missing", "Borr_State_GDP_Missing", "Proj_State_GDP_Missing", "Missing_Borr_State_Income", "Missing_Proj_State_Income", "Missing_Borr_State_Vacancy", "Missing_Proj_State_Vacancy", "Missing_Borr_State_Homeowner", "Missing_Proj_State_Homeowner"]
    
    norm_columns(data, columns_to_normalize)

def normalize_partial(data):
    """
    Removing potential problem columns (missing data)
    """
    columns_to_remove = ['BorrState Income', 'ProjState Income', 'Missing Borr Income', 'Missing Proj Income','BorrState GDP', 'ProjState GDP', 'Missing Borr GDP', 'Missing Proj GDP']
    columns_to_remove += ['CORPORATION', 'INDIVIDUAL', 'Missing Business']
    remove_columns(data, columns_to_remove)

    columns_to_normalize = ['ThirdPartyDollars', 'GrossApproval', 'TermInMonths', 'Avg Home Price', 'GDP Delta YR', 'Log S&P Open', 'BorrState Unemployment', 'ProjectState Unemployment', 'BorrState Vacancy', 'ProjectState Vacancy', 'Unemployment YR']
    norm_columns(data, columns_to_normalize)

def main():
    file_name = "data_sandbox/combined_updated.csv"
    data = read_csv_file(file_name)

    columns_to_remove = ['Idx', 'BorrState', 'BorrZip', 'ProjectState', 'subpgmdesc', 'DeliveryMethod', 'BusinessType', 'NaicsCode', 'ApprovalDate', 'ChargeOffDate', 'GrossChargeOffAmount', 'ApprovalFiscalYear']

    remove_columns(data, columns_to_remove)
    change_labels(data)

    normalize_all(data)
    # normalize_partial(data)

    add_intercept(data)
    write_csv_file(data, f'{file_name[:-4]}_norm_full.csv')

if __name__ == "__main__":
    main()