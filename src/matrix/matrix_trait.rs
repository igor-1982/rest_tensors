use std::ops::Range;

pub struct IncreaseStepBy<I> {
    pub iter: I,
    step: usize,
    increase: usize,
    first_take: bool, 
}

impl<I> IncreaseStepBy<I> {
    pub fn new(iter: I, step: usize, increase: usize) -> IncreaseStepBy<I> {
        assert!(step!=0);
        IncreaseStepBy {iter, step, first_take: true, increase }
    }
}

impl<I> Iterator for IncreaseStepBy<I> 
where I: Iterator,
{
    type Item = I::Item;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.first_take {
            self.first_take = false;
            //self.step -= self.increase;
            self.iter.next()
        } else {
            let cur_step = self.step;
            self.step += self.increase;
            self.iter.nth(cur_step)
        }
    }
}



pub struct SubMatrixStepBy<I> {
    pub iter: I,
    rows: Range<usize>,
    columns: Range<usize>,
    size: [usize;2],
    step: usize,
    max: usize,
    position: usize,
    first_take: bool,
}
impl<I> SubMatrixStepBy<I> {
    pub fn new(iter: I, rows: Range<usize>, columns: Range<usize>, size:[usize;2]) -> SubMatrixStepBy<I> {
        let position =columns.start*size[0] + rows.start;
        let step = size[0]-rows.end+rows.start;
        let max = (columns.end-1)*size[0] + rows.end-1;
        SubMatrixStepBy{iter, rows, columns, size, step, position,max,first_take: true}
    }
}


impl<I> Iterator for SubMatrixStepBy<I>
where I:Iterator,
{
    type Item = I::Item;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {

        // MARK:: change by Igor 24-05-11, need double check
        let curr_row = self.position%unsafe{self.size.get_unchecked(0)};
        //let curr_column = self.position/unsafe{self.size.get_unchecked(0)};

        let is_in_row_range = curr_row >= self.rows.start && curr_row < self.rows.end;
        //let is_in_col_range = curr_column >= self.columns.start && curr_column < self.columns.end;
        
        //let is_in_range = curr_row >= self.rows.start && curr_row < self.rows.end &&
        //                        curr_column >= self.columns.start && curr_column < self.columns.end;
        if self.position > self.max {
            None
        } else if self.first_take {
            self.position += 1;
            self.first_take = false;
            self.iter.nth(self.position-1)
        } else if is_in_row_range {
            //self.step -= self.increase;
            self.position += 1;
            self.iter.next()
        } else {
            self.position += self.step+1;
            self.iter.nth(self.step)
        }
    }
}

pub struct SubMatrixInUpperStepBy<I> {
    pub iter: I,
    rows: Range<usize>,
    columns: Range<usize>,
    size: [usize;2],
    step: usize,
    max: Option<usize>,
    position: usize,
    first_take: bool,
}
impl<I> SubMatrixInUpperStepBy<I> {
    pub fn new(iter: I, rows: Range<usize>, columns: Range<usize>, size:[usize;2]) -> SubMatrixInUpperStepBy<I> {
        //let position =columns.start*size[0] + rows.start;
        let step = size[0]-rows.end+rows.start;
        //let max = (columns.end-1)*size[0] + rows.end-1;
        let position =if rows.start<=columns.start {
            columns.start*size[0] + rows.start
            //columns.start*(columns.start+1)/2 + rows.start
        } else {
            rows.start*size[0] + rows.start
        };
        let max = if rows.start>columns.end-1 {
            None
        } else if columns.end >= rows.end {
            Some((columns.end-1)*size[0] + rows.end-1)
            //Some((columns.end-1)*columns.end/2 + rows.end-1)
        } else {
            Some((columns.end-1)*size[0] + columns.end-1)
            //Some((columns.end-1)*columns.end/2 + columns.end-1)
        };
        SubMatrixInUpperStepBy{iter, rows, columns, size, step, position,max,first_take: true}
    }
}


impl<I> Iterator for SubMatrixInUpperStepBy<I>
where I:Iterator,
{
    type Item = I::Item;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {

        let curr_row = self.position%unsafe{self.size.get_unchecked(0)};
        let curr_column = self.position/unsafe{self.size.get_unchecked(0)};

        let is_in_row_range = curr_row >= self.rows.start && curr_row < self.rows.end;
        
        //let is_in_range = 
        //    curr_row >= self.rows.start && curr_row < self.rows.end &&
        //    curr_column >= self.columns.start && curr_column < self.columns.end;

        let is_in_upper = curr_row <= curr_column;

        if let Some(max) = self.max {
            if self.position > max {
                None
            } else if self.first_take {
                self.position += 1;
                self.first_take = false;
                self.iter.nth(self.position-1)
            } else if is_in_row_range {
                if is_in_upper {
                    //self.step -= self.increase;
                    self.position += 1;
                    self.iter.next()
                } else {
                    let step  = (curr_column+1)*self.size[0] + self.rows.start - self.position;
                    self.position += step + 1; 
                    self.iter.nth(step)
                }
            } else {
                self.position += self.step+1;
                self.iter.nth(self.step)
            }
        } else {
            None
        }
    }
}

pub struct MatrixUpperStepBy<I> {
    pub iter: I,
    size: [usize;2],
    step: usize,
    position: usize,
    first_take: bool,
}

impl<I> MatrixUpperStepBy<I> {
    pub fn new(iter: I, size:[usize;2]) -> MatrixUpperStepBy<I> {
        let position =0;
        let step = size[0];
        MatrixUpperStepBy{iter, size, step, position,first_take: true}
    }
    pub fn new_shift(iter: I, size:[usize;2], shift: usize) -> MatrixUpperStepBy<I> {
        let position =shift;
        let step = size[0];
        MatrixUpperStepBy{iter, size, step, position,first_take: true}
    }
}

impl<I> Iterator for MatrixUpperStepBy<I>
where I:Iterator,
{
    type Item = I::Item;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {

        let curr_row = self.position%unsafe{self.size.get_unchecked(0)};
        let curr_column = self.position/unsafe{self.size.get_unchecked(0)};
        
        let is_in_range = curr_row <= curr_column;

        if self.first_take {
            self.position = 1;
            self.first_take = false;
            self.iter.next()
        } else if is_in_range {
            //self.step -= self.increase;
            self.position += 1;
            self.iter.next()
        } else {
            let step = self.size[0]-curr_column;
            self.position += step;
            self.iter.nth(step-1)
        }
    }
}

pub trait MatrixIterator: Iterator {
    type Item;
    fn step_by_increase(self, step:usize, increase: usize) -> IncreaseStepBy<Self>
    where Self:Sized {
        IncreaseStepBy::new(self, step, increase)
    }
    //pub fn new(iter: I, rows: Range<usize>, columns: Range<usize>, size:[usize;2]) -> SubMatrixStepBy<I> {
    fn submatrix_step_by(self, rows: Range<usize>, columns: Range<usize>, size:[usize;2]) -> SubMatrixStepBy<Self> 
    where Self:Sized {
        SubMatrixStepBy::new(self, rows, columns, size)
    }
    fn matrixupper_step_by(self, size:[usize;2]) -> MatrixUpperStepBy<Self> 
    where Self:Sized {
        MatrixUpperStepBy::new(self, size)
    }
    fn submatrix_in_upper_step_by(self, rows: Range<usize>, columns: Range<usize>, size:[usize;2]) -> SubMatrixInUpperStepBy<Self> 
    where Self:Sized {
        SubMatrixInUpperStepBy::new(self, rows, columns, size)
    }
    fn matrixupper_step_by_shift(self, size:[usize;2], shift: usize) -> MatrixUpperStepBy<Self> 
    where Self:Sized {
        MatrixUpperStepBy::new_shift(self, size, shift)
    }
}

impl<'a,T> MatrixIterator for std::slice::Iter<'a,T> {
    type Item = T;
}
impl<'a,T> MatrixIterator for std::slice::IterMut<'a,T> {
    type Item = T;
}