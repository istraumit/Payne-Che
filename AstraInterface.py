import numpy as np
from astra.database.astradb import Task, DataProduct
from astra.base import TaskInstance, Parameter, DictParameter
from run_MWMPayne import fit_BOSS


class AstraInterface(TaskInstance):
    NN_path = Parameter(bundled=True)
    data_format = Parameter(bundled=True)
    log_dir = Parameter(bundled=True)

    wave_range = Parameter()
    spectral_R = Parameter()
    N_chebyshev = Parameter(default=15)
    N_presearch_iter = Parameter(default=1)
    N_presearch = Parameter(default=4000)


    def pre_execute(self):
        # Iterate over the tasks (either one task or a bundle)
        # and check that the object types are as we expect.
        for task, input_data_products, parameters in self.iterable():

            # Here, the "task" variable is a database record
            assert isinstance(task, Task)

            # The input_data_products is a list of DataProduct objects for this task
            assert isinstance(input_data_products, list)
            for data_product in input_data_products:
                assert isinstance(data_product, DataProduct)

            # The "parameters" is a dictionary of parameters for this task.
            assert isinstance(parameters, dict)

        self.opt_list = ['NN_path', 'wave_range', 'spectral_R', 'N_chebyshev', 'N_presearch_iter', 'N_presearch', 'data_format', 'log_dir']
        self.db_field_list = ['teff', 'u_teff', 'logg', 'u_logg', 'vsini', 'u_vsini', 'v_micro', 'u_v_micro', 'm_h', 'u_m_h', 'v_rad', 'u_v_rad']
        return None


    def get_opt_dict(self):
        opt = {}
        for pn in self.opt_list:
            opt[pn] = getattr(self, pn)
        return opt


    def execute(self):
        opt = self.get_opt_dict()

        NN = Network()
        NN.read_in(self.NN_path)

        logger = FitLoggerDB(self.log_dir)
        logger.init_DB()
        logger.new_run(str(opt))

        loader = SpectrumLoader(self.data_format)

        results = []
        for task, input_data_products, parameters in self.iterable():
            for dp in input_data_products:
                sp = loader.get_single(dp.path)
                sd = sp.load()

                SNR, db_values, db_cheb = fit_BOSS(sd, NN, parameters, logger)
                result = {}
                result['snr'] = SNR
                for i,name in enumerate(self.db_field_list):
                    result[name] = db_values[i]
                result['theta'] = db_cheb

                # here 'result' is a dict of results for this task, where each key will correspond to a field name in the `PayneCheOutputs` table
                with database.atomic():
                    task.create_or_update_outputs('thepayne_che', [result])

                results.append(res)

        return results
















