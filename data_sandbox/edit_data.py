import csv
import numpy as np

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

def add_SP500(data):
    month_data = read_csv_file("SP500_Month_Return.csv")
    yr_data = read_csv_file("SP500_YR_Return.csv")

    data[0].append("SP500_Past_Month_Return")
    for row in data[1:]:
        date = row[5]
        for i in range(1, len(month_data)):
            if int(month_data[i][0]) == int(date[:4]):
                if int(month_data[i][1]) == int(date[5:7]):
                    row.append(month_data[i][2])
                    break
                
    data[0].append("SP500_Past_YR_Return")
    for row in data[1:]:
        date = row[5]
        for i in range(1, len(yr_data)):
            if int(yr_data[i][0]) + 1 == int(date[:4]):
                row.append(yr_data[i][1])
                break 
            

def add_home_price(data):
    home_data = read_csv_file("Median_New_Home_Price.csv")

    data[0].append("US_Med_Home_Price_Month_Change")
    data[0].append("US_Med_Home_Price_YR_Change")
    for row in data[1:]:
        date = row[5]
        for i in range(1, len(home_data)):
            if home_data[i][0][:7] == date[:7]:
                row.append(home_data[i][2])
                row.append(home_data[i][3])
                break
            
def add_unemployment(data):
    un_data = read_csv_file("Unemployment.csv")

    data[0].append("Borr_State_Unemployment_YR_Change")
    data[0].append("Borr_State_Unemployment_Missing")
    for row in data[1:]:
        borr = row[1]
        if borr == "GU" or borr == "VI" or borr == "PW":
            row.append(0)
            row.append(1)
        else:
            yr = row[5][:4]
            idx = un_data[0].index(yr)
            for i in range(1, len(un_data)):
                if un_data[i][0] == borr:
                    row.append(un_data[i][idx])
                    row.append(0)
                    break

    data[0].append("Proj_State_Unemployment_YR_Change")
    data[0].append("Proj_State_Unemployment_Missing")
    for row in data[1:]:
        proj = row[11]
        if proj == "GU" or proj == "VI" or proj == "PW":
            row.append(0)
            row.append(1)
        else:
            yr = row[5][:4]
            idx = un_data[0].index(yr)
            for i in range(1, len(un_data)):
                if un_data[i][0] == proj:
                    row.append(un_data[i][idx])
                    row.append(0)
                    break
            
def add_gdp(data):
    gdp_data = read_csv_file("StateGDP.csv")

    data[0].append("Borr_State_GDP_YR_Change")
    data[0].append("Borr_State_GDP_Missing")
    for row in data[1:]:
        yr = row[5][:4]
        borr = row[1]
        if int(yr) < 1997 or borr == "GU" or borr == "VI" or borr == "PW":
            row.append(0)
            row.append(1)
        else:       
            borr_idx = gdp_data[0].index(borr)
            for i in range(1, len(gdp_data)):
                if gdp_data[i][0] == yr:
                    row.append(gdp_data[i][borr_idx])
                    row.append(0)
                    break
                
    data[0].append("Proj_State_GDP_YR_Change")
    data[0].append("Proj_State_GDP_Missing")
    for row in data[1:]:
        yr = row[5][:4]
        proj = row[1]
        if int(yr) < 1997 or proj == "GU" or proj == "VI" or proj == "PW":
            row.append(0)
            row.append(1)
        else:       
            proj_idx = gdp_data[0].index(proj)
            for i in range(1, len(gdp_data)):
                if gdp_data[i][0] == yr:
                    row.append(gdp_data[i][proj_idx])
                    row.append(0)
                    break                

def add_income(data):
    income_data = read_csv_file("StateIncome.csv")

    data[0].append("Borr_State_Income_YR_Change")
    data[0].append("Missing_Borr_State_Income")
    for row in data[1:]:
        yr = row[5][:4]
        borr = row[1]
        if borr == "PR" or borr == "GU" or borr == "VI" or borr == "PW":
            row.append(0)
            row.append(1)
        else:
            borr_idx = income_data[0].index(borr)
            for i in range(1, len(income_data)):
                if income_data[i][0][:4] == yr:
                    row.append(income_data[i][borr_idx])
                    row.append(0)
                    break

    
    data[0].append("Proj_State_Income_YR_Change")
    data[0].append("Missing_Proj_State_Income")
    for row in data[1:]:
        yr = row[5][:4]
        proj = row[11]
        if proj == "PR" or proj == "GU" or proj == "VI" or proj == "PW":
            row.append(0)
            row.append(1)
        else:
            proj_idx = income_data[0].index(proj)
            for i in range(1, len(income_data)):
                if income_data[i][0][:4] == yr:
                    row.append(income_data[i][proj_idx])
                    row.append(0)
                    break
            
def add_vacancy(data):
    vac_data = read_csv_file("Vacancy.csv")

    data[0].append("Borr_State_Vacancy_YR_Change")
    data[0].append("Missing_Borr_State_Vacancy")
    for row in data[1:]:
        yr = row[5][:4]
        borr = row[1]
        if borr == "PR" or borr == "GU" or borr == "VI" or borr == "PW":
            row.append(0)
            row.append(1)
        else:
            borr_idx = vac_data[0].index(borr)
            for i in range(1, len(vac_data)):
                if vac_data[i][0][:4] == yr:
                    row.append(vac_data[i][borr_idx])
                    row.append(0)
                    break
    
    data[0].append("Proj_State_Vacancy_YR_Change")
    data[0].append("Missing_Proj_State_Vacancy")
    for row in data[1:]:
        yr = row[5][:4]
        proj = row[1]
        if proj == "PR" or proj == "GU" or proj == "VI" or proj == "PW":
            row.append(0)
            row.append(1)
        else:
            proj_idx = vac_data[0].index(proj)
            for i in range(1, len(vac_data)):
                if vac_data[i][0][:4] == yr:
                    row.append(vac_data[i][proj_idx])
                    row.append(0)
                    break
            

def add_homeowner(data):
    homeowner_data = read_csv_file("Homeownership.csv")

    data[0].append("Borr_State_Homeowner_YR_Change")
    data[0].append("Missing_Borr_State_Homeowner")
    for row in data[1:]:
        yr = row[5][:4]
        borr = row[1]
        if borr == "PR" or borr == "GU" or borr == "VI" or borr == "PW":
            row.append(0)
            row.append(1)
        else:
            borr_idx = homeowner_data[0].index(borr)
            for i in range(1, len(homeowner_data)):
                if homeowner_data[i][0][:4] == yr:
                    row.append(homeowner_data[i][borr_idx])
                    row.append(0)
                    break
    
    data[0].append("Proj_State_Homeowner_YR_Change")
    data[0].append("Missing_Proj_State_Homeowner")
    for row in data[1:]:
        yr = row[5][:4]
        proj = row[1]
        if proj == "PR" or proj == "GU" or proj == "VI" or proj == "PW":
            row.append(0)
            row.append(1)
        else:
            proj_idx = homeowner_data[0].index(proj)
            for i in range(1, len(homeowner_data)):
                if homeowner_data[i][0][:4] == yr:
                    row.append(homeowner_data[i][proj_idx])
                    row.append(0)
                    break

def add_fed(data):
    fed_data = read_csv_file("FEDFUNDS.csv")

    data[0].append("Fed_Fund_Month_Change")
    data[0].append("Fed_Fund_YR_Change")
    for row in data[1:]:
        date = row[5]
        for i in range(1, len(fed_data)):
            if fed_data[i][0][:7] == date[:7]:
                row.append(fed_data[i][2])
                row.append(fed_data[i][3])
                break
            

def add_indicators(data):
    ind_data = read_csv_file("Indicators.csv")

    data[0].append("CCI_Month_Change")
    data[0].append("BCI_Month_Change")
    data[0].append("CLI_Month_Change")
    data[0].append("CCI_YR_Change")
    data[0].append("BCI_YR_Change")
    data[0].append("CLI_YR_Change")
    for row in data[1:]:
        date = row[5]
        for i in range(1, len(ind_data)):
            if ind_data[i][0] == date[:7]:
                row.append(ind_data[i][4])
                row.append(ind_data[i][5])
                row.append(ind_data[i][6])
                row.append(ind_data[i][7])
                row.append(ind_data[i][8])
                row.append(ind_data[i][9])
                break

def main():
    file_name = "combined.csv"
    data = read_csv_file(file_name)

    columns_to_remove = ["SP500 YR", "Unemployment YR", "Avg Home Price", "GDP Delta YR", "BorrState Unemployment","ProjectState Unemployment", "BorrState Income", "ProjState Income", "Missing Borr Income", "Missing Proj Income", "BorrState GDP", "ProjState GDP", "Missing Borr GDP", "Missing Proj GDP", "BorrState Vacancy", "ProjectState Vacancy"]
    remove_columns(data, columns_to_remove)

    add_SP500(data)
    add_home_price(data)
    add_unemployment(data)
    add_gdp(data)
    add_income(data)
    add_vacancy(data)
    add_homeowner(data)
    add_fed(data)
    add_indicators(data)

    write_csv_file(data, f'{file_name[:-4]}_updated.csv')

if __name__ == "__main__":
    main()