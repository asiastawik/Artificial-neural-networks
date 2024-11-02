import numpy as np
from epf_arx import epf_arx

data = np.loadtxt('GEFCOM.txt')
startd = 0   # First day of the calibration window (startd from Matlab minus 1)
endd = 360   # First day to forecast (equal to endd from Matlab)
Ndays = 180  # user provided number of days to be predicted #540-361
             # (max is 722 for GEFCom with endd=360)

from datetime import datetime
start_time = datetime.now()

# naive, arx, narx or narx_mc
forecast_type = 'narx'

#TASK 1
# # Estimate and compute forecasts of the ARX model
# res = epf_arx(data[:, :4], Ndays, startd, endd, forecast_type)
# np.savetxt(f'res_{forecast_type.lower()}_{i}.txt', res)
#
# end_time = datetime.now()
# print('Duration: {}'.format(end_time - start_time))
#
# # Compute and display MAE
# print(f'MAE for days {endd + 1} to {endd + Ndays} across all hours')
# print(f'(length of the calibration window for point forecasts = {endd} days)')
# print(f'{forecast_type} MAE: {np.mean(np.abs(res[:, 2] - res[:, 3]))}')

#Task 2
#zmień number of neurons na 5
#w forecast.py -> units     hidden = Dense(units=1, activation='sigmoid')(inputs)# Hidden layer (5 neurons; GM = 20)
for i in range(1, 11):
    # Estimate and compute forecasts of the ARX model
    res = epf_arx(data[:, :4], Ndays, startd, endd, forecast_type)
    np.savetxt(f'res_{forecast_type.lower()}_{i}.txt', res)

    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))

mae_list = []
for i in range(1, 11):
    file_list = []
    for j in range(1, i+1+1):
        files = [np.loadtxt(f'res_{forecast_type.lower()}_{j}.txt')]
        for file in files:
            file_list.append(file[:, -1])
    averages = np.mean(file_list, axis=0)
    mae_list.append(np.mean(np.abs(averages - file[:, -2])))

print(mae_list)
print(len(mae_list))

import matplotlib as plt
plt.plot(range(1,11), mae_list)
plt.ylabel("Errors")
plt.xlabel("n")
plt.show()



# Compute and display MAE
print(f'MAE for days {endd+1} to {endd+Ndays} across all hours')
print(f'(length of the calibration window for point forecasts = {endd} days)')
print(f'{forecast_type} MAE: {np.mean(np.abs(res[:, 2] - res[:, 3]))}')
