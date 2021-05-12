#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import lib.mfa as mfa
import linecache

class MolecularFieldAnalysis:
    def __init__(self, mean_plane = 5, origin = 2, x_axis = 5, n = [0,1,2,3,4,6], grid_x=6, grid_y=8, grid_z=8, threshold=0.01):
        self.mean_plane = mean_plane
        self.origin = origin
        self.x_axis = x_axis
        self.n = n
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = grid_z
        self.threshold = threshold
        self.unitcell_size = 1

    def indicator_field(self, mol_name):
        mol_name_train = np.loadtxt(mol_name,dtype='str')
        Y = self._standard_xyz(mol_name)
        mol_field = mfa.indicator_field(self.grid_x,self.grid_y,self.grid_z,self.unitcell_size)
        N_samples = len(mol_name_train)
        for i in range(0,N_samples):
            Atoms = np.loadtxt(mol_name_train[i],skiprows=2,usecols=(0,),unpack=True,dtype='str')
            Z = self._align_xyz(mol_name, i, Y)
            N_atoms = self._atom_number(mol_name, i)
            indicator_field = mol_field.calc_field(Z,Atoms,N_atoms)
            if i == 0:
                indicator_fields = indicator_field
            else:
                indicator_fields = np.c_[indicator_fields,indicator_field]
        indicator_fields = indicator_fields.T
        descriptor = mol_field.prescreening(N_samples,indicator_fields)
        return (indicator_fields, descriptor)

    def indicator_field_sub(self, mol_name, Y):
        mol_name_train = np.loadtxt(mol_name,dtype='str')
        mol_field = mfa.indicator_field(self.grid_x,self.grid_y,self.grid_z,self.unitcell_size)
        N_samples = len(mol_name_train)
        for i in range(0,N_samples):
            Atoms = np.loadtxt(mol_name_train[i],skiprows=2,usecols=(0,),unpack=True,dtype='str')
            Z = self._align_xyz(mol_name, i, Y)
            N_atoms = self._atom_number(mol_name, i)
            indicator_field = mol_field.calc_field(Z,Atoms,N_atoms)
            if i == 0:
                indicator_fields = indicator_field
            else:
                indicator_fields = np.c_[indicator_fields,indicator_field]
        indicator_fields = indicator_fields.T
        descriptor = mol_field.prescreening(N_samples,indicator_fields)
        return (indicator_fields, descriptor)

    def indicator_field_test(self, mol_name,mol_name_pred):
        indicator_fields = self.indicator_field(mol_name)
        indicator_fields = indicator_fields[0]
        mol_name_test = np.loadtxt(mol_name_pred,dtype='str')
        Y = self._standard_xyz(mol_name)
        mol_field = mfa.indicator_field(self.grid_x,self.grid_y,self.grid_z,self.unitcell_size)
        N_samples = len(mol_name_test)
        for i in range(0,N_samples):
            mol_name = mol_name_test[i]
            Atoms = np.loadtxt(mol_name,skiprows=2,usecols=(0,),unpack=True,dtype='str')
            Z = self._align_xyz(mol_name_test, i, Y)
            N_atoms = self._atom_number(mol_name_test, i)
            indicator_field_test = mol_field.calc_field(Z,Atoms,N_atoms)
            if i == 0:
                indicator_fields_test = indicator_field_test
            else:
                indicator_fields_test = np.c_[indicator_fields_test,indicator_field_test]
        indicator_fields_test = indicator_fields_test.T
        descriptor = mol_field.prescreening_test(N_samples,indicator_fields,indicator_fields_test)
        return (indicator_fields_test, descriptor)

    def indicator_field_corr(self, mol_name,target_variable):
        mol_name_train = np.loadtxt(mol_name,dtype='str')
        Y = self._standard_xyz(mol_name)
        mol_field = mfa.indicator_field(self.grid_x,self.grid_y,self.grid_z,self.unitcell_size)
        N_samples = len(mol_name_train)
        for i in range(0,N_samples):
            Atoms = np.loadtxt(mol_name_train[i],skiprows=2,usecols=(0,),unpack=True,dtype='str')
            Z = self._align_xyz(mol_name, i, Y)
            N_atoms = self._atom_number(mol_name, i)
            indicator_field = mol_field.calc_field(Z,Atoms,N_atoms)
            if i == 0:
                indicator_fields = indicator_field
            else:
                indicator_fields = np.c_[indicator_fields,indicator_field]
        indicator_fields = indicator_fields.T
        descriptor_pre = mol_field.prescreening(N_samples,indicator_fields)
        j = 0
        n = len(descriptor_pre[:,0])-1
        for i in range(len(descriptor_pre[0,:])):
            corr = (np.corrcoef(descriptor_pre[1:n+1,i],target_variable))
            if np.abs(corr[1,0]) > 0.3:
                if j == 0:
                    descriptor = descriptor_pre[:,i]
                    j += 1
                else:
                    descriptor = np.c_[descriptor,descriptor_pre[:,i]]
        return (indicator_fields, descriptor)

    def indicator_field_test_sub(self, mol_name,mol_name_pred, Y):
        indicator_fields = self.indicator_field_sub(mol_name,Y)
        indicator_fields = indicator_fields[0]
        mol_name_test = np.loadtxt(mol_name_pred,dtype='str')
        mol_field = mfa.indicator_field(self.grid_x,self.grid_y,self.grid_z,self.unitcell_size)
        N_samples = len(mol_name_test)
        for i in range(0,N_samples):
            mol_name = mol_name_test[i]
            Atoms = np.loadtxt(mol_name,skiprows=2,usecols=(0,),unpack=True,dtype='str')
            Z = self._align_xyz(mol_name_test, i, Y)
            N_atoms = self._atom_number(mol_name_test, i)
            indicator_field_test = mol_field.calc_field(Z,Atoms,N_atoms)
            if i == 0:
                indicator_fields_test = indicator_field_test
            else:
                indicator_fields_test = np.c_[indicator_fields_test,indicator_field_test]
        indicator_fields_test = indicator_fields_test.T
        descriptor = mol_field.prescreening_test(N_samples,indicator_fields,indicator_fields_test)
        return (indicator_fields_test, descriptor)

    def indicator_field_corr_test(self, mol_name,target_variable,sample_number):
        mol_name_train = np.loadtxt(mol_name,dtype='str')
        Y = self._standard_xyz(mol_name)
        mol_field = mfa.indicator_field(self.grid_x,self.grid_y,self.grid_z,self.unitcell_size)
        N_samples = len(mol_name_train)
        for i in range(0,N_samples):
            Atoms = np.loadtxt(mol_name_train[i],skiprows=2,usecols=(0,),unpack=True,dtype='str')
            Z = self._align_xyz(mol_name, i, Y)
            N_atoms = self._atom_number(mol_name, i)
            indicator_field = mol_field.calc_field(Z,Atoms,N_atoms)
            if i == 0:
                indicator_fields = indicator_field
            else:
                indicator_fields = np.c_[indicator_fields,indicator_field]
        indicator_fields = indicator_fields.T
        j = 0
        for i in range(len(indicator_fields[0,:])):
            corr = (np.corrcoef(indicator_fields[0:sample_number,i],target_variable))
            if np.abs(corr[1,0]) > 0.3:
                if j == 0:
                    descriptor = indicator_fields[:,i]
                    j += 1
                else:
                    descriptor = np.c_[descriptor,indicator_fields[:,i]]
        return (indicator_fields, descriptor)

    def structure_info(self, coefficient,mol_name,sample_number,indicator_fields):
        mol_field = mfa.indicator_field(self.grid_x,self.grid_y,self.grid_z,self.unitcell_size)
        coordinate = mol_field.unitcell_coordinate()
        coef = np.loadtxt(coefficient, skiprows=1, usecols=(1,),unpack=True)
        Vis = mol_field.visualization_preprocess(coordinate, indicator_fields, coef)
        important_coordinate = mol_field.structual_information_coordinate(Vis, self.threshold)
        mol_name_train = np.loadtxt(mol_name,dtype='str')
        Y = self._standard_xyz(mol_name)
        mol_field = mfa.indicator_field(self.grid_x,self.grid_y,self.grid_z,self.unitcell_size)
        Atoms = np.loadtxt(mol_name_train[sample_number],skiprows=2,usecols=(0,),unpack=True,dtype='str')
        Z = self._align_xyz(mol_name, sample_number, Y)
        N_atoms = self._atom_number(mol_name, sample_number)
        indicator_field = mol_field.calc_field(Z,Atoms,N_atoms)
        indicator_fields = indicator_fields.T
        important_information = mol_field.structual_information(indicator_field, Vis, self.threshold)
        return (important_coordinate,important_information)

    def structure_info_sub(self, coefficient,mol_name,sample_number,indicator_fields, Y):
        mol_field = mfa.indicator_field(self.grid_x,self.grid_y,self.grid_z,self.unitcell_size)
        coordinate = mol_field.unitcell_coordinate()
        coef = np.loadtxt(coefficient, skiprows=1, usecols=(1,),unpack=True)
        Vis = mol_field.visualization_preprocess(coordinate, indicator_fields, coef)
        important_coordinate = mol_field.structual_information_coordinate(Vis, self.threshold)
        mol_name_train = np.loadtxt(mol_name,dtype='str')
        mol_field = mfa.indicator_field(self.grid_x,self.grid_y,self.grid_z,self.unitcell_size)
        Atoms = np.loadtxt(mol_name_train[sample_number],skiprows=2,usecols=(0,),unpack=True,dtype='str')
        Z = self._align_xyz(mol_name, sample_number, Y)
        N_atoms = self._atom_number(mol_name, sample_number)
        indicator_field = mol_field.calc_field(Z,Atoms,N_atoms)
        indicator_fields = indicator_fields.T
        important_information = mol_field.structual_information(indicator_field, Vis, self.threshold)
        return (important_coordinate,important_information)

    def structure_info_corr(self, coefficient,mol_name,sample_number,indicator_fields,target_variable):
        N_samples = len(target_variable)
        mol_field = mfa.indicator_field(self.grid_x,self.grid_y,self.grid_z,self.unitcell_size)
        coordinate = mol_field.unitcell_coordinate()
        coef = np.loadtxt(coefficient, skiprows=1, usecols=(1,),unpack=True)
        Vis = mol_field.visualization_preprocess_corr(coordinate, indicator_fields, coef, target_variable,N_samples)
        important_coordinate = mol_field.structual_information_coordinate(Vis, self.threshold)
        mol_name_train = np.loadtxt(mol_name,dtype='str')
        Y = self._standard_xyz(mol_name)
        mol_field = mfa.indicator_field(self.grid_x,self.grid_y,self.grid_z,self.unitcell_size)
        Atoms = np.loadtxt(mol_name_train[sample_number],skiprows=2,usecols=(0,),unpack=True,dtype='str')
        Z = self._align_xyz(mol_name, sample_number, Y)
        N_atoms = self._atom_number(mol_name, sample_number)
        indicator_field = mol_field.calc_field(Z,Atoms,N_atoms)
        indicator_fields = indicator_fields.T
        important_information = mol_field.structual_information(indicator_field, Vis, self.threshold)
        return (important_coordinate,important_information)

    def xyz_file(self,mol_name,sample_number,important_information,Directory):
        mol_name_train = np.loadtxt(mol_name,dtype='str')
        Y = self._standard_xyz(mol_name)
        Z = self._align_xyz(mol_name, sample_number, Y)
        N_atoms = self._atom_number(mol_name, sample_number)
        mol_name = mol_name_train[sample_number]
        mv_name = Directory + mol_name.replace("data/","")
        atoms = np.loadtxt(mol_name,skiprows=2,usecols=(0,),unpack=True,dtype='S3')
        coordinate = important_information[0]
        label = important_information[1]
        n_all = N_atoms + len(important_information[1])
        with open(mv_name,'w') as f:
            f.write(str(n_all)+'\n'+mv_name+'\n')
            for i in range(N_atoms):
                f.write(str((atoms.T[i]).decode())+'\t'+str(Z[i,0])+'\t'+str(Z[i,1])+'\t'+str(Z[i,2])+'\n')
            for i in range(len(important_information[1])):
                f.write(str(label[i])+'\t'+str(coordinate[i,0])+'\t'+str(coordinate[i,1])+'\t'+str(coordinate[i,2])+'\n')

    def xyz_file_sub(self,mol_name,sample_number,important_information,Directory, Y):
        mol_name_train = np.loadtxt(mol_name,dtype='str')
        Z = self._align_xyz(mol_name, sample_number, Y)
        N_atoms = self._atom_number(mol_name, sample_number)
        mol_name = mol_name_train[sample_number]
        mv_name = Directory + mol_name.replace("data/","")
        atoms = np.loadtxt(mol_name,skiprows=2,usecols=(0,),unpack=True,dtype='S3')
        coordinate = important_information[0]
        label = important_information[1]
        n_all = N_atoms + len(important_information[1])
        with open(mv_name,'w') as f:
            f.write(str(n_all)+'\n'+mv_name+'\n')
            for i in range(N_atoms):
                f.write(str((atoms.T[i]).decode())+'\t'+str(Z[i,0])+'\t'+str(Z[i,1])+'\t'+str(Z[i,2])+'\n')
            for i in range(len(important_information[1])):
                f.write(str(label[i])+'\t'+str(coordinate[i,0])+'\t'+str(coordinate[i,1])+'\t'+str(coordinate[i,2])+'\n')

    def prediction(self, coefficient,mol_name,sample_number,indicator_fields):
        mol_field = mfa.indicator_field(self.grid_x,self.grid_y,self.grid_z,self.unitcell_size)
        coordinate = mol_field.unitcell_coordinate()
        intercept =np.loadtxt(coefficient, usecols=(1,),unpack=True)
        intercept = intercept[0]
        coef = np.loadtxt(coefficient, skiprows=1, usecols=(1,),unpack=True)
        coef = mol_field.visualization_preprocess(coordinate, indicator_fields, coef)
        mol_name_train = np.loadtxt(mol_name,dtype='str')
        Y = self._standard_xyz(mol_name)
        mol_field = mfa.indicator_field(self.grid_x,self.grid_y,self.grid_z,self.unitcell_size)
        Atoms = np.loadtxt(mol_name_train[sample_number],skiprows=2,usecols=(0,),unpack=True,dtype='str')
        Z = self._align_xyz(mol_name, sample_number, Y)
        N_atoms = self._atom_number(mol_name, sample_number)
        indicator_field = mol_field.calc_field(Z,Atoms,N_atoms)
        pred_value = mol_field.prediction(indicator_field, coef) + intercept
        return (pred_value)

    def R2q2(self,outcomes):
        measured,pred,pred_LOOCV = np.loadtxt(outcomes+"_output.csv",skiprows=1, usecols=(1,2,3,),unpack=True,delimiter=",")
        R2 = 1-sum((measured-pred)**2)/sum((measured-np.average(measured))**2)
        q2 = 1-sum((measured-pred_LOOCV)**2)/sum((measured-np.average(measured))**2)
        print (outcomes,"R2:",round(R2,3),"q2:",round(q2,3))

    def _standard_xyz(self,mol_name):
        mol_name_train = np.loadtxt(mol_name,dtype='str')
        X = np.loadtxt(mol_name_train[0],skiprows=2,usecols=(1,2,3))
        N_atoms = linecache.getline(mol_name_train[0], int(1))
        N_atoms = int(N_atoms)
        mol = mfa.alignment(N_atoms,self.mean_plane,self.origin,self.x_axis,self.n)
        Y = mol.def_plane(X)
        return Y

    def _align_xyz(self,mol_name,sample_number,Y):
        mol_name_train = np.loadtxt(mol_name,dtype='str')
        N_atoms = linecache.getline(mol_name_train[0], int(1))
        N_atoms = int(N_atoms)
        mol = mfa.alignment(N_atoms,self.mean_plane,self.origin,self.x_axis,self.n)
        mol_name = mol_name_train[sample_number]
        Z = np.loadtxt(mol_name,skiprows=2,usecols=(1,2,3))
        N_atoms = linecache.getline(mol_name, int(1))
        N_atoms = int(N_atoms)
        Z = mol.align_mol(Y, Z, N_atoms)
        return Z


    def _atom_number(self,mol_name,sample_number):
        mol_name_train = np.loadtxt(mol_name,dtype='str')
        mol_name = mol_name_train[sample_number]
        N_atoms = linecache.getline(mol_name, int(1))
        N_atoms = int(N_atoms)
        return N_atoms