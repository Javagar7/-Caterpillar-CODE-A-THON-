failure_dates[part] = current_time + pd.Timedelta(seconds=(i+1) * time_step_seconds)
                break