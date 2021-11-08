#![allow(unused)]
extern crate blas;
extern crate blas_src;
extern crate lapack;
extern crate lapack_src;

use std::fmt::Display;
use std::ops::{Add, AddAssign, Sub, SubAssign};

use lapack::{dsyev,dspevx,dspgvx};
use blas::dgemm;
mod tensors_slice;
use crate::tensors_slice::TensorsSlice;

const SAFE_MINIMUM:f64 = 10E-12;

#[derive(Clone, Copy,Debug, PartialEq)]
pub enum MatFormat {
    Full,
    Upper,
    Lower
}

//recursive wrapper
struct RecFn<T>(Box<dyn Fn(&RecFn<T>,(T,T)) -> (T,T)>);
impl<T> RecFn<T> {
    fn call(&self, f: &RecFn<T>, n: (T,T)) -> (T,T) {
        (self.0(f,n))
    }
}

pub trait Indexing {
    fn indexing(&self, position:&[usize]) -> usize;
    fn reverse_indexing(&self, position:usize) -> Vec<usize>;
}

#[derive(Clone,Debug,PartialEq)]
pub struct Tensors<T:Clone+Display> {
    /// Coloum-major Tensors designed for quantum chemistry calculations specifically.
    pub store_format : MatFormat,
    pub rank: usize,
    pub size : Vec<usize>,
    pub indicing: Vec<usize>,
    pub data : Vec<T>
}

impl <T: Clone + Display> Tensors<T> {
    pub fn new(new_type: String, size: Vec<usize>, new_default: T) -> Tensors<T> {
        let mut mat_format = if new_type.to_lowercase()==String::from("full") {MatFormat::Full
        } else if new_type.to_lowercase()==String::from("upper") {MatFormat::Upper
        } else if new_type.to_lowercase()==String::from("lower") {MatFormat::Lower
        } else {
            panic!("Error in determing the layout format of the matrix: {}", new_type)
        };
        if size.len()==1 {
            mat_format = match mat_format {
                MatFormat::Full => {MatFormat::Full},
                MatFormat::Upper => {
                    println!("WARNNING: a simple 1-D array is allocated, which cannot be stored as a upper matrix");
                    MatFormat::Full
                },
                MatFormat::Lower => {
                    println!("WARNNING: a simple 1-D array is allocated, which cannot be stored as a lower matrix");
                    MatFormat::Full
                },
            }
        } else if size.len()>=2 && size[0]!=size[1] {
            mat_format = match mat_format {
                MatFormat::Full => {MatFormat::Full},
                MatFormat::Upper => {
                    println!("WARNNING: a {}-D tensor ({:?}) is allocated, which cannot be stored as a upper matrix ", size.len(),&size);
                    MatFormat::Full
                },
                MatFormat::Lower => {
                    println!("WARNNING: a {}-D tensor ({:?}) is allocated, which cannot be stored as a lower matrix", size.len(),&size);
                    MatFormat::Full
                },
            }
        }

        let mut len: usize = 1;
        let mut indicing: Vec<usize> = vec![0;size.len()];
        match &mat_format {
            MatFormat::Full => {
                for i in 0..size.len() {
                    indicing[i]=len;
                    len *= size[i];
                };
            },
            MatFormat::Lower => {
                len = size[0]*(size[0]+1)/2;
                for i in 2..size.len() {
                    indicing[i]=len;
                    len *= size[i];
                };
            },
            MatFormat::Upper => {
                len = size[0]*(size[0]+1)/2;
                indicing[1]=len;
                for i in 2..size.len() {
                    indicing[i]=len;
                    len *= size[i];
                };
            }
        }
        Tensors {
            store_format: mat_format,
            rank: size.len(),
            size,
            indicing,
            data: vec![new_default.clone(); len]
        }
    }
    pub fn from_vec(new_type: String, size: Vec<usize>, new_vec: Vec<T>) -> Tensors<T> {
        let mut mat_format = if new_type.to_lowercase()==String::from("full") {MatFormat::Full
        } else if new_type.to_lowercase()==String::from("upper") {MatFormat::Upper
        } else if new_type.to_lowercase()==String::from("lower") {MatFormat::Lower
        } else {
            panic!("Error in determing the layout format of the matrix: {}", new_type)
        };
        if size.len()==1 {
            mat_format = match mat_format {
                MatFormat::Full => {MatFormat::Full},
                MatFormat::Upper => {
                    println!("WARNNING: a simple 1-D array is allocated, which cannot be stored as a upper matrix");
                    MatFormat::Full
                },
                MatFormat::Lower => {
                    println!("WARNNING: a simple 1-D array is allocated, which cannot be stored as a lower matrix");
                    MatFormat::Full
                },
            }
        } else if size.len()>=2 && size[0]!=size[1] {
            mat_format = match mat_format {
                MatFormat::Full => {MatFormat::Full},
                MatFormat::Upper => {
                    println!("WARNNING: a {}-D tensor is allocated, which cannot be stored as a upper matrix ", size.len());
                    MatFormat::Full
                },
                MatFormat::Lower => {
                    println!("WARNNING: a {}-D tensor is allocated, which cannot be stored as a lower matrix", size.len());
                    MatFormat::Full
                },
            }
        }
        let mut len: usize = 1;
        let mut indicing: Vec<usize> = vec![0;size.len()];
        match &mat_format {
            MatFormat::Full => {
                for i in 0..size.len() {
                    indicing[i]=len;
                    len *= size[i];
                };
            },
            MatFormat::Lower => {
                len = size[0]*(size[0]+1)/2;
                for i in 2..size.len() {
                    indicing[i]=len;
                    len *= size[i];
                };
            },
            MatFormat::Upper => {
                len = size[0]*(size[0]+1)/2;
                for i in 2..size.len() {
                    indicing[i]=len;
                    len *= size[i];
                };
            }
        }
        if len>new_vec.len() {
            panic!("Error: inconsistency happens when formating a tensor from a given vector, (length from size, length of new vector) = ({},{})",len,new_vec.len());
        } else if len<new_vec.len() {
            println!("Waring: the vector size ({}) is larger for the size of the new tensor ({})", new_vec.len(), len);
        }
        Tensors {
            store_format: mat_format,
            rank: size.len(),
            size,
            indicing,
            data: new_vec
        }
    }
    pub fn duplicate(&self, out_type: String)  -> Option<Tensors<T>> {
        let mut new_type = if out_type.to_lowercase()==String::from("full") {MatFormat::Full
        } else if out_type.to_lowercase()==String::from("upper") {MatFormat::Upper
        } else if out_type.to_lowercase()==String::from("lower") {MatFormat::Lower
        } else if out_type.to_lowercase()==String::from("unchange") {self.store_format.clone()
        } else {
            panic!("Error in determing the layout format of the matrix: {}", out_type)
        };
        let mut self_type = match &self.store_format {
            MatFormat::Full => String::from("full"),
            MatFormat::Upper => String::from("upper"),
            MatFormat::Lower => String::from("lower")
        };
        let new_size = self.size.clone();
        let new_default: T = self.data[0].clone();
        //if self_type == out_type_2 {
        let mut new_tensor = if self.store_format == new_type {
            let mut tmp_tensor = Tensors::new(self_type,new_size,new_default);
            tmp_tensor.data = self.data.clone();
            tmp_tensor
        } else {
            let mut tmp_tensor = Tensors::new(out_type,new_size,new_default);
            let tmp_size_vec = 0;
            (0..self.data.len()).into_iter().for_each(|i| {
                let mut tensor_position = self.reverse_indexing(i);
                tmp_tensor.set(&tensor_position,self.data[i].clone());
                if let MatFormat::Full = tmp_tensor.store_format {
                    let tmp_switch = tensor_position[0];
                    tensor_position[0] = tensor_position[1];
                    tensor_position[1] = tmp_switch;
                    tmp_tensor.set(&tensor_position,self.data[i].clone());
                }
            });
            tmp_tensor
        };
        Some(new_tensor)

    }
    pub fn mat_shape(&mut self, start_dimention: usize) -> Option<(usize,usize)> {
        if let Some(i_dim) = self.size.get(start_dimention) {
            if let Some(j_dim) = self.size.get(start_dimention+1) {
                Some((*i_dim,*j_dim))
            } else {
                None
            }
        } else {
            None
        }
    }
    pub fn get(&self, position: &[usize]) -> Option<T> {
        if position.len()!=self.rank {
            println!("Error: it is a {}-D tensor, which cannot be indexed by a position of {:?}", self.rank,position);
            None
        } else {
            let index: usize = self.indexing(position);
            Some(self.data[index].clone())
        }
    }
    pub fn get_mut(&mut self, position: &[usize]) -> Option<&mut T>{
        if position.len()!=self.rank {
            println!("Error: it is a {}-D tensor, which cannot be indexed by a position of {:?}", self.rank,position);
            None
        } else {
            let index: usize = self.indexing(position);
            Some(&mut self.data[index])
        }
    }
    pub fn set(&mut self, position: &[usize], new_data: T){
        if position.len()!=self.rank {
            panic!("Error: it is a {}-D tensor, which cannot be indexed by a position of {:?}", self.rank,position);
        } else {
            let index: usize = self.indexing(position);
            self.data[index] = new_data;
        }
    }
    pub fn get_reducing_tensor_mut(&mut self, i_reduced: usize) -> Option<TensorsSlice<T>> {
        if i_reduced > self.size[self.size.len()-1] {
            None
        } else {
            let mut position:Vec<usize> = vec![0; self.rank];
            position[self.rank-1] = i_reduced;
            let p_start = self.indexing(&position);
            let p_length = self.indicing[self.rank-1];
            Some(TensorsSlice {
                store_format: self.store_format.clone(),
                rank: self.rank-1,
                size: self.size[..self.rank-1].to_vec(),
                indicing: self.indicing[..self.rank-1].to_vec(),
                data : &mut self.data[p_start..p_start+p_length]})
        }
    }
    pub fn get_reducing_tensor(&mut self, i_reduced: usize) -> Option<Tensors<T>> {
        if i_reduced > self.size[self.size.len()-1] {
            None
        } else {
            let mut position:Vec<usize> = vec![0; self.rank];
            position[self.rank-1] = i_reduced;
            let p_start = self.indexing(&position);
            let p_length = self.indicing[self.rank-1];
            Some(Tensors {
                store_format: self.store_format.clone(),
                rank: self.rank-1,
                size: self.size[..self.rank-1].to_vec(),
                indicing: self.indicing[..self.rank-1].to_vec(),
                data : self.data[p_start..p_start+p_length].to_vec().clone()})
        }
    }
    pub fn to_tensorsslice(&mut self) -> Option<TensorsSlice<T>> {
        Some(TensorsSlice {
            store_format: self.store_format.clone(),
            rank: self.rank,
            size: self.size.clone(),
            indicing: self.indicing.clone(),
            data : &mut self.data})
        
    }
    pub fn get_reducing_matrix(&mut self, i_reduced: Vec<usize>) -> Option<TensorsSlice<T>> {
        if self.rank<2 || (self.rank > 2 && self.rank - i_reduced.len()!=2) {
            println!("Error:: the tensor cannot be reduced to a matrix");
            None
        } else if (self.size[0]!=self.size[1]) {
            println!("Error:: the first and second dimensions of the tensor have different length, thus it cannot be reduced to a matrix");
            None
        } else if self.rank>2 {
            let mut position:Vec<usize> = vec![0; self.rank];
            (0..i_reduced.len()).into_iter().for_each(|i| {
                let p= position.get_mut(i+2).unwrap();
                *p=i_reduced[i];
            });
            let p_start = 0;
            let p_length = self.indicing[2];
            Some(TensorsSlice {
                store_format: self.store_format.clone(),
                rank: 2,
                size: self.size[..self.rank-1].to_vec(),
                indicing: self.indicing[..2].to_vec(),
                data : &mut self.data[p_start..p_start+p_length]})
        } else {
            Some(TensorsSlice {
                store_format: self.store_format.clone(),
                rank: 2,
                size: self.size.clone(),
                indicing: self.indicing[..2].to_vec(),
                data : &mut self.data})
        }
    }

    pub fn formated_output(&mut self, n_len: usize, mat_form: String) {
        let mat_format = if mat_form.to_lowercase()==String::from("full") {MatFormat::Full
        } else if mat_form.to_lowercase()==String::from("upper") {MatFormat::Upper
        } else if mat_form.to_lowercase()==String::from("lower") {MatFormat::Lower
        } else {
            panic!("Error in determing the layout format of the matrix: {}", mat_form)
        };
        if self.rank!=2 {panic!("At present, the formated output is only available for the 2-D matrix")};
        let n_row = self.size[0];
        let n_column = self.size[1];
        let n_block = if n_column%n_len==0 {n_column/n_len} else {n_column/n_len+1};
        let mut index:usize = 0;
        //println!("{}",n_block);
        (0..n_block).into_iter().for_each(|i_block| {
            let t_len = if (i_block+1)*n_len<=n_column {n_len} else {n_column%n_len};
            //println!("{},{}",i_block,t_len);
            let mut tmp_s:String = format!("{:5}","");
            for i in 0..t_len {
                if tmp_s.len()==5 {
                    tmp_s = format!("{} {:12}",tmp_s,i+i_block*n_len);
                } else {
                    tmp_s = format!("{},{:12}",tmp_s,i+i_block*n_len);
                }
            }
            println!("{}",tmp_s);
            for i in 0..n_row as usize {
                let mut tmp_s = format!("{:5}",i);
                let j_start = i_block*n_len;
                let mut turn_off_comma = true;
                for j in (j_start..j_start+t_len) {
                    match &mat_format {
                        MatFormat::Full => {
                            let tmp_f = self.get(&[i,j]).unwrap();
                            if tmp_s.len()==5 {
                                tmp_s = format!("{} {:12.6}",tmp_s,tmp_f);
                            } else {
                                tmp_s = format!("{},{:12.6}",tmp_s,tmp_f);
                            }
                        },
                        MatFormat::Upper => {
                            if i<=j {
                                let tmp_f = self.get(&[i,j]).unwrap();
                                if turn_off_comma {
                                    tmp_s = format!("{} {:12.6}",tmp_s,tmp_f);
                                    turn_off_comma = false;
                                } else {
                                    tmp_s = format!("{},{:12.6}",tmp_s,tmp_f);
                                }
                            } else {
                                tmp_s = format!("{} {:12}",tmp_s,String::from(" "));
                            }
                        },
                        MatFormat::Lower => {
                            if i>=j {
                                let tmp_f = self.get(&[i,j]).unwrap();
                                if tmp_s.len()==5 {
                                    tmp_s = format!("{} {:12.6}",tmp_s,tmp_f);
                                } else {
                                    tmp_s = format!("{},{:12.6}",tmp_s,tmp_f);
                                }
                            }
                        }
                    };
                    //println!("{},{}",tmp_i,j);
                };
                if tmp_s.len()>5 {println!("{}",tmp_s)};
            }
        });
    }
}

impl <T: Clone+Display+Add<Output=T>+AddAssign> Add for Tensors<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        let mut new_tensors = self.duplicate("unchange".to_string()).unwrap();
        (0..new_tensors.data.len()).into_iter().for_each(|i| {
            new_tensors.data[i] += other.data[i].clone();
        });
        new_tensors
    }
}

impl <T: Clone+Display+Sub<Output=T>+SubAssign> Sub for Tensors<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        let mut new_tensors = self.duplicate("unchange".to_string()).unwrap();
        (0..new_tensors.data.len()).into_iter().for_each(|i| {
            new_tensors.data[i] -= other.data[i].clone();
        });
        new_tensors
    }
}

impl <T: Clone+Display> Indexing for Tensors<T> {
    fn indexing(&self, positions: &[usize]) -> usize 
        where T: Clone
    {
        let mut p_start: usize = 0;
        match self.store_format {
            MatFormat::Full => {
                let mut i:usize = 0;
                for i in 0..self.rank {
                    p_start += self.indicing[i]*positions[i];
                }
            },
            MatFormat::Lower => {
                let mut i: usize = 0;
                p_start = if positions[0]>=positions[1] {
                    (2*self.size[0]-positions[1]-1)*positions[1]/2 + positions[0]
                } else {
                    (2*self.size[0]-positions[0]-1)*positions[0]/2 + positions[1]
                };
                for i in 2..self.rank {
                    p_start += self.indicing[i]*positions[i];
                }
            },
            MatFormat::Upper => {
                let mut i: usize = 0;
                p_start = if positions[0]<=positions[1] {
                    (positions[1]+1)*positions[1]/2 + positions[0]
                } else {
                    (positions[0]+1)*positions[0]/2 + positions[1]
                };
                for i in 2..self.rank {
                    p_start += self.indicing[i]*positions[i];
                }
            }
        }
        p_start
    }
    fn reverse_indexing(&self, positions: usize) -> Vec<usize> 
        where T: Clone
    {
        let mut pos: Vec<usize> = vec![0;self.size.len()];
        let mut rest = positions;
        match self.store_format {
            MatFormat::Full => {
                (0..self.size.len()).rev().for_each(|i| {
                    pos[i] = rest/self.indicing[i];
                    rest = rest % self.indicing[i];
                })
            },
            MatFormat::Lower => {
                // according the definition of lower tensor, the first and second ranks have the same lenght
                (2..self.size.len()).rev().for_each(|i|{
                    pos[i] = rest/self.indicing[i];
                    rest = rest % self.indicing[i];
                });
                let index_pattern = RecFn(Box::new(|func: &RecFn<i32>, n: (i32,i32)| -> (i32,i32) {
                    match n.0 {
                        i if i<0 => {
                            (0,n.1+1)
                        },
                        0=>(0,n.1),
                        _ => {func.call(func,(n.0-n.1, n.1-1))}
                    }
                }));
                let (i,j) = index_pattern.call(&index_pattern,(rest as i32,self.size[0] as i32));
                let i = rest - (self.size[0]+j as usize -1)*(self.size[0]-j as usize)/2;
                let j = self.size[0] - j as usize;
                pos[0] = i;
                pos[1] = j as usize;
            },
            MatFormat::Upper => {
                // according the definition of lower tensor, the first and second ranks have the same lenght
                (2..self.size.len()).rev().for_each(|i|{
                    pos[i] = rest/self.indicing[i];
                    let rest = rest % self.indicing[i];
                });
                let index_pattern = RecFn(Box::new(|func: &RecFn<i32>, n: (i32,i32)| -> (i32,i32) {
                    match n.0-n.1 {
                        i if i<0 => {
                            (0,n.1-1)
                        },
                        0 => (0,n.1),
                        _ => {func.call(func,(n.0-n.1, n.1+1))}
                    }
                }));
                let (i,j) = index_pattern.call(&index_pattern,(rest as i32,0));
                let tmp_val = (j*(j+1)/2);
                let i = rest - tmp_val as usize;
                pos[0] = i;
                pos[1] = j as usize;
            },
        }
        pos
    }
}

impl Tensors<f64> {
    pub fn dot(&mut self, b: &mut Tensors<f64>) -> Option<Tensors<f64>> {
        let mut tmp_a = self.to_tensorsslice().unwrap();
        let mut tmp_b = b.to_tensorsslice().unwrap();
        Some(tmp_a.dot(&mut tmp_b).unwrap())
    }
    pub fn abs(&mut self) -> f64 {
        let mut result = 0.0;
        (0..self.data.len()).into_iter().for_each(|i|{
            result += self.data[i].powf(2.0);
        });
        result.sqrt()
    }
    pub fn multiple(&mut self, scaled_factor: f64) {
        let mut tmp_a = self.to_tensorsslice().unwrap();
        tmp_a.multiple(scaled_factor);
    }
    pub fn diagonalize(&mut self) -> Option<(Tensors<f64>,Vec<f64>,i32)> {
        let mut tmp_a = self.to_tensorsslice().unwrap();
        tmp_a.diagonalize()
    }
    pub fn lapack_solver(&mut self,ovlp:&mut Tensors<f64>,num_orb:usize) -> Option<(Tensors<f64>,Vec<f64>)> {
        let mut tmp_a = self.to_tensorsslice().unwrap();
        let mut tmp_b = ovlp.to_tensorsslice().unwrap();
        tmp_a.lapack_solver(tmp_b, num_orb)
    }
    pub fn transpose(&mut self) -> Option<Tensors<f64>> {
        match self.store_format {
            MatFormat::Full => {
                let tmp_size = self.size.clone();
                let mut new_ten: Tensors<f64> = Tensors::new(String::from("full"), tmp_size, 0.0);
                (0..self.size[0]).into_iter().for_each(|i| {
                    (0..self.size[1]).into_iter().for_each(|j| {
                        new_ten.set(&[i,j],self.get(&[j,i]).unwrap());
                    })
                });
                Some(new_ten)
            },
            MatFormat::Lower => {
                let tmp_size = self.size.clone();
                let mut new_ten: Tensors<f64> = Tensors::new(String::from("upper"), tmp_size, 0.0);
                (0..self.size[0]).into_iter().for_each(|i| {
                    (0..i+1).into_iter().for_each(|j| {
                        new_ten.set(&[j,i],self.get(&[i,j]).unwrap());
                    })
                });
                Some(new_ten)
            },
            MatFormat::Upper => {
                let tmp_size = self.size.clone();
                let mut new_ten: Tensors<f64> = Tensors::new(String::from("lower"), tmp_size, 0.0);
                (0..self.size[0]).into_iter().for_each(|i| {
                    (0..i+1).into_iter().for_each(|j| {
                        new_ten.set(&[i,j],self.get(&[j,i]).unwrap());
                    })
                });
                Some(new_ten)
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use crate::{Indexing, Tensors};
    #[test]
    fn test_upper_matrix() {
        let size:Vec<usize>=vec![11,11];
        let mut tmp_v = vec![0.0;(size[1]+1)*size[1]/2];
        (0..tmp_v.len()).into_iter().for_each(|i| {
            tmp_v[i] = i as f64;
        });
        //let size = vec![11,11];
        let mut my_mat = Tensors::from_vec(String::from("upper"), size, tmp_v);
        let mut tmp_v = if let Some(fvalue) = my_mat.get_mut(&[2,6]) {
            fvalue
        } else {
            panic!("Error in getting a matrix element");
        };
        *tmp_v = 100.0;
        println!("index: {}",&my_mat.indexing(&[2,6]));
        println!("pos:   {:?}",&my_mat.reverse_indexing(0));
        println!("value: {:?}",&my_mat.get(&[2,6]).unwrap());
        //my_mat.formated_output(5, String::from("upper"));
        //assert_eq!(my_mat.get(&[2,6]).unwrap(), 100.0);
    }
    #[test]
    fn test_upper_tensor() {
        let size:Vec<usize>=vec![11,11,2];
        let mut tmp_v = vec![0.0;(size[0]+1)*size[1]/2*size[2]];
        (0..tmp_v.len()).into_iter().for_each(|i| {
            tmp_v[i] = i as f64;
        });
        let mut my_mat = Tensors::from_vec(String::from("upper"), size, tmp_v);
        let mut tmp_v = if let Some(fvalue) = my_mat.get_mut(&[2,6,1]) {
            fvalue
        } else {
            panic!("Error in getting a matrix element");
        };
        *tmp_v = 100.0;
        println!("{:?}",&my_mat);
        //my_mat.formated_output(5, String::from("full"));
        let mut tmp_rd_mat = my_mat.get_reducing_tensor(1).unwrap();
        tmp_rd_mat.formated_output(5, String::from("upper"));
        tmp_rd_mat.set(&[2,6],50.0);
        tmp_rd_mat.formated_output(5, String::from("upper"));
        assert_eq!(my_mat.get(&[2,6,1]).unwrap(), 50.0);
    }
    #[test]
    fn test_dgemm() {
        let size_a:Vec<usize>=vec![2,3];
        let mut tmp_a = vec![
            1.0,4.0,
            2.0,5.0,
            3.0,6.0];
        let size_b:Vec<usize>=vec![3,4];
        let mut tmp_b = vec![
            1.0,5.0, 9.0,
            2.0,6.0,10.0,
            3.0,7.0,11.0,
            4.0,8.0,11.0];
        let mut my_a = Tensors::from_vec(String::from("full"), size_a, tmp_a);
        let mut my_b = Tensors::from_vec(String::from("full"), size_b, tmp_b);
        let mut my_c = my_a.dot(&mut my_b).unwrap();
        //my_c.formated_oupput
        my_c.formated_output(5,String::from("full"));
        println!("{:?}", my_c);
        //assert_eq!(my_c.get(&[2,6,1]).unwrap(), 50.0);
    }
    #[test]
    fn test_diagonalize() {
        let size_a:Vec<usize>=vec![3,3];
        let mut tmp_a = vec![
            3.0,1.0,1.0,
            1.0,3.0,1.0,
            1.0,1.0,3.0];
        let mut my_a = Tensors::from_vec("full".to_string(), size_a, tmp_a);
        let (mut eigenvectors,mut eigenvalues,mut n) = my_a.diagonalize().unwrap();
        my_a.formated_output(5,String::from("full"));
        println!("eigenvalues: {:?}",eigenvalues);
        eigenvectors.formated_output(5,String::from("full"));
        let mut my_b = my_a.dot(&mut eigenvectors).unwrap();
        my_b.formated_output(5,String::from("full"));
        let mut eigenvectors_trans = eigenvectors.transpose().unwrap();
        let mut my_c = eigenvectors_trans.dot(&mut my_b).unwrap();
        my_c.formated_output(5,String::from("full"));

        let mut tmp_a = vec![
            3.0,1.0,3.0,1.0,1.0,3.0
            ];
        let size_a:Vec<usize>=vec![3,3];
        let mut my_a = Tensors::from_vec("upper".to_string(), size_a, tmp_a);
        let (mut full_eigenvectors,mut eigenvalues,mut n) = my_a.diagonalize().unwrap();
        println!("{:?}",eigenvalues);
        full_eigenvectors.formated_output(5,String::from("full"));
        //let mut full_eigenvectors = eigenvectors.copy(String::from("full")).unwrap();
        let mut full_my_a = my_a.duplicate(String::from("full")).unwrap();
        full_my_a.formated_output(5,String::from("full"));
        //full_eigenvectors.formated_output(5,String::from("full"));
        let mut my_b = full_my_a.dot(&mut full_eigenvectors).unwrap();
        let mut full_eigenvectors_trans = full_eigenvectors.transpose().unwrap();
        let mut my_c = full_eigenvectors_trans.dot(&mut my_b).unwrap();
        my_c.formated_output(5,String::from("full"));
    }
}