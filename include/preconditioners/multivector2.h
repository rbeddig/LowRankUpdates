/**
 * @file multivector2.h
 * @author Rebekka Beddig
 * @version 0.1
 */
#ifndef MULTIVECTOR2
#define MULTIVECTOR2

// deal.II
#include <deal.II/base/exceptions.h>
#include <deal.II/base/subscriptor.h>

#include <deal.II/lac/lapack_templates.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector_memory.templates.h>


// Trilinos
#include <Epetra_FEVector.h>
#include <Epetra_SerialComm.h>
#include <Epetra_Vector.h>

// STL
#include <algorithm>
#include <memory>
#include <random>
#include <vector>

namespace dealii
{
  namespace Preconditioners
  {
    /**
     * @brief MultiVector class that handles (thin) rectangular matrices. The data is stored in an Epetra_FEVector.
     */
    class MultiVector2 : public Subscriptor
    {
    public:
      using VectorType = TrilinosWrappers::MPI::Vector;
      using MatrixType = TrilinosWrappers::SparseMatrix;

      using value_type      = double;
      using real_type       = TrilinosScalar;
      using size_type       = types::global_dof_index;
      using iterator        = value_type *;
      using const_iterator  = const value_type *;
      using reference       = value_type &;
      using const_reference = const value_type &;

      /**
       * Points to a single column of @ref MultiVector2 and can be used to calculate with this column.
       *
       */
      class Column : public Subscriptor
      {
      public:
        using value_type      = double;
        using real_type       = TrilinosScalar;
        using size_type       = types::global_dof_index;
        using iterator        = value_type *;
        using const_iterator  = const value_type *;
        using reference       = value_type &;
        using const_reference = const value_type &;

        /**
         * Default constructor.
         */
        Column()
          : Subscriptor()
          , view(false)
        {}

        /**
         * Constructor.
         * @param src rectangular matrix
         * @param index number of the needed column
         * @param copyview copy or view the column [Copy/View]
         */
        Column(const Epetra_MultiVector &src,
               const int                 index,
               const Epetra_DataAccess & copyview = View)
          : Subscriptor()
        {
          column.reset();
          column = std::make_unique<Epetra_Vector>(copyview, src, index);
          // column = std::make_unique<Epetra_Vector>(*((src)(index)));
          view = (copyview == View ? true : false);
        }

        /**
         * Constructor.
         * @param src rectangular matrix
         * @param index number of the needed column
         * @param copyview copy or view the column [Copy/View]
         */
        Column(const MultiVector2 &     src,
               const int                index,
               const Epetra_DataAccess &copyview = View)
          : Subscriptor()
        {
          column.reset();
          column =
            std::make_unique<Epetra_Vector>(copyview, *(src.vector), index);
          view = (copyview == View ? true : false);
        }
        /**
         * Constructor
         */
        Column(const Epetra_BlockMap &map)
          : Subscriptor()
        {
          column.reset();
          column = std::make_unique<Epetra_Vector>(map);
          view   = false;
        }

        /**
         * @brief Copy constructor
         */
        Column(const Column &col)
          : Subscriptor()
        {
          column.reset();
          column = std::make_unique<Epetra_Vector>(col.column->Map(), true);
          view   = false;
        }

        /**
         * @brief Move constructor
         *
         */
        Column(Column &&col) noexcept
          : Column()
        {
          std::swap(this->column, col.column);
          std::swap(this->view, col.view);
        }

        /**
         * @brief Constructor
         *
         */
        Column(const unsigned int size)
          : Column()
        {
          reinit(size);
        }


        /**
         * Destructor
         */
        ~Column() override = default;


        /**
         * Set column of MultiVector2
         */
        void
        set_column(MultiVector2 &mv, const unsigned int index) const
        {
          *(mv.trilinos_vector()(index)) = *(column.get());
        }


        /**
         * Sets all elements to zero.
         */
        bool
        all_zero() const
        {
          // loop over all the elements
          int          index = 0;
          unsigned int flag  = 0;
          while (index != size())
            {
              if ((*column)[index] != 0)
                {
                  flag = 1;
                  break;
                }
              ++index;
            }

          return flag == 0;
        }


        /**
         * Resize the current object to have the same size and layout as
         * the model_vector argument provided. The second argument
         * indicates whether to clear the current object after resizing.
         * The second argument must have a default value equal to false.
         */
        void
        reinit(const Column &model_vector,
               const bool    leave_elements_uninitialized = false)
        {
          if (view)
            column.release();
          column.reset();
          // set size
          column =
            std::make_unique<Epetra_Vector>(model_vector.column->Map(),
                                            !leave_elements_uninitialized);
          view = false;
        }

        /** View the @param index-th column of @param src. */
        void
        reinit(const MultiVector2 &     src,
               const int                index,
               const Epetra_DataAccess &copyview = View)
        {
          if (view)
            column.release();
          column.reset();
          column =
            std::make_unique<Epetra_Vector>(copyview, *(src.vector), index);
          view = true;
        }

        void
        reinit(const unsigned int size)
        {
          if (view)
            column.release();
          column.reset();
          Epetra_BlockMap map((int)size,
                              1,
                              0,
                              Utilities::Trilinos::comm_self());
          column = std::make_unique<Epetra_Vector>(map, true /*zero out*/);
          view   = false;
        }

        /**
         * Inner product between the current object and @ref v.
         * Computes result = this * v.
         *
         * @param[in] v Vector
         * @return result = this * v.
         */
        double
        operator*(const Column &v) const
        {
          // Assert(column->Map().SameAs(v.column->Map()),
          //        ExcDifferentParallelPartitioning());

          double result;

          const int ierr = column->Dot(*(v.column), &result);
          AssertThrow(ierr == 0, ExcTrilinosError(ierr));

          return result;
        }


        /**
         * Inner product between the current object and @ref v.
         * Computes result = this * v.
         *
         * @param[in] v Vector
         * @return result = this * v.
         */
        void
        operator+=(const Column &v)
        {
          this->add(1.0, v);
        }

        /**
         * Inner product between the current object and @ref v.
         * Computes result = this * v.
         *
         * @param[in] v Vector
         * @return result = this * v.
         */
        void
        operator-=(const Column &v)
        {
          this->add(-1.0, v);
        }



        /**
         * Scale the matrix column (describes a multiplication with a diagonal
         * matrix)
         *
         * @param[in] scaling scaling vector
         */
        void
        scale(const TrilinosWrappers::MPI::Vector &scaling)
        {
          const int ierr =
            column->Multiply(1.0, scaling.trilinos_vector(), *column, 0.0);
          AssertThrow(ierr == 0, ExcTrilinosError(ierr));
        }



        /**
         * Access the element @ref idx of the column.
         */
        double &
        operator[](const int idx)
        {
          return column->operator[](idx);
        }

        /**
         * Access the element @ref idx of the column.
         */
        const double &
        operator[](const int idx) const
        {
          return column->operator[](idx);
        }

        /**
         * Access the element @ref idx of the column.
         */
        double &
        operator()(const int idx)
        {
          return column->operator[](idx);
        }

        /**
         * Access the element @ref idx of the column.
         */
        const double &
        operator()(const int idx) const
        {
          return column->operator[](idx);
        }

        /**
         * Sets all the vector entries to a constant scalar.
         */
        Column &
        operator=(const double s)
        {
          AssertIsFinite(s);

          int ierr = column->PutScalar(s);
          AssertThrow(ierr == 0, ExcTrilinosError(ierr));

          return *this;
        }

        /**
         * Deep copy of the vector.
         * Important if Column contains pointers to data to duplicate data.
         */
        Column &
        operator=(const Column &v)
        {
          Assert(column.get() != nullptr,
                 ExcMessage("Vector is not constructed properly."));

          // check equality for MPI communicators to avoid accessing a possibly
          // invalid MPI_Comm object
#ifdef DEAL_II_WITH_MPI
          const Epetra_MpiComm *my_comm =
            dynamic_cast<const Epetra_MpiComm *>(&column->Comm());
          const Epetra_MpiComm *v_comm =
            dynamic_cast<const Epetra_MpiComm *>(&v.column->Comm());
          const bool same_communicators =
            my_comm != nullptr && v_comm != nullptr &&
            my_comm->DataPtr() == v_comm->DataPtr();
          // Need to ask MPI whether the communicators are the same. We would
          // like to use the following checks but currently we cannot make sure
          // the memory of my_comm is not stale from some MPI_Comm_free
          // somewhere. This can happen when a vector lives in
          // GrowingVectorMemory data structures. Thus, the following code is
          // commented out.
          //
          // if (my_comm != nullptr &&
          //     v_comm != nullptr &&
          //     my_comm->DataPtr() != v_comm->DataPtr())
          //  {
          //    int communicators_same = 0;
          //    const int ierr = MPI_Comm_compare (my_comm->GetMpiComm(),
          //                                       v_comm->GetMpiComm(),
          //                                       &communicators_same);
          //    AssertThrowMPI(ierr);
          //    if (!(communicators_same == MPI_IDENT ||
          //          communicators_same == MPI_CONGRUENT))
          //      same_communicators = false;
          //    else
          //      same_communicators = true;
          //  }
#else
          const bool same_communicators = true;
#endif

          // distinguish three cases. First case: both vectors have the same
          // layout (just need to copy the local data, not reset the memory and
          // the underlying Epetra_Map). The third case means that we have to
          // rebuild the calling vector.
          if (same_communicators && v.column->Map().SameAs(column->Map()))
            {
              *column = *v.column;
            }
          // Second case: vectors have the same global
          // size, but different parallel layouts (and
          // one of them a one-to-one mapping). Then we
          // can call the import/export functionality.
          // else if (size() == v.size() &&
          //          (v.column->Map().UniqueGIDs() ||
          //          column->Map().UniqueGIDs()))
          //   {
          //     reinit(v, false, true);
          //   }
          // Third case: Vectors do not have the same
          // size.
          else
            {
              column = std::make_unique<Epetra_Vector>(*v.column);
            }


          return *this;
        }


        /** write Vector<double> to Preconditioners::MultiVector2::Column */
        Column &
        operator=(const Vector<double> &v)
        {
          Assert(size() == (int)v.size(),
                 ExcDimensionMismatch(size(), v.size()));

          // this is probably not very efficient but works. in particular,  we
          // could do better if we know that number==TrilinosScalar because then
          // we could elide the copying of elements
          //
          // let's hope this isn't a particularly frequent operation
          for (int i = 0; i < column->MyLength(); ++i)
            (*column)[i] = v(i);

          return *this;
        }

        /**
         * Returns the size of the column.
         */
        int
        size() const
        {
          return column->GlobalLength();
        }

        /**
         * Returns the size of the column.
         */
        int
        n_vectors() const
        {
          return 1;
        }
        /**
         * Returns the underlying Epetra_Vector that stores the data.
         */
        Epetra_Vector &
        trilinos_vector()
        {
          return *column;
        }


        /**
         * Returns the underlying Epetra_Vector that stores the data.
         */
        const Epetra_MultiVector &
        trilinos_vector() const
        {
          return static_cast<const Epetra_MultiVector &>(*column);
        }

        /**
         * Adds the column to the vector @param v.
         * *this = (*this) + v
         */
        void
        add(const Column &v)
        {
          this->add(1.0, v);
        }

        /**
         * Computes the scaled addition of vectors
         * *this = (*this) + a * v
         *
         * @param a scalar
         * @param v vector
         */
        void
        add(const double a, const Column &v)
        {
          AssertIsFinite(a);

          const int ierr = column->Update(a, *(v.column), 1.);
          AssertThrow(ierr == 0, ExcTrilinosError(ierr));
        }

        /**
         * Computes the scaled addition of vectors
         * *this = s * (*this) + a * v
         *
         * @param s scalar
         * @param a scalar
         * @param v vector
         */
        void
        sadd(const double s, const double a, const Column &v)
        {
          Assert(size() == v.size(), ExcDimensionMismatch(size(), v.size()));
          AssertIsFinite(s);
          AssertIsFinite(a);

          const int ierr = column->Update(a, *(v.column), s);
          AssertThrow(ierr == 0, ExcTrilinosError(ierr));
        }

        /**
         * Scaled assignment of a vector *this = a*v.
         */
        void
        equ(const double a, const Column &v)
        {
          AssertIsFinite(a);

          // If we don't have the same map, copy.
          if (column->Map().SameAs(v.column->Map()) == false)
            {
              this->sadd(0., a, v);
            }
          else
            {
              // Otherwise, just update
              int ierr = column->Update(a, *v.column, 0.0);
              AssertThrow(ierr == 0, ExcTrilinosError(ierr));
            }
        }

        /**
         * Combined scaled addition of vector x into the current object and
         * subsequent inner product of the current object with v.
         * Computes
         *     *this += a * v,
         * followed by
         *     return_value = (*this) + w.
         */
        double
        add_and_dot(const double a, const Column &V, const Column &W)
        {
          this->add(a, V);
          return *this * W;
        }

        /**
         * Multiply the elements of the current object by a fixed value.
         * *this *= a
         */
        Column &
        operator*=(const double a)
        {
          AssertIsFinite(a);

          const int ierr = column->Scale(a);
          AssertThrow(ierr == 0, ExcTrilinosError(ierr));

          return *this;
        }

        /**
         * Returns the l2 norm of the vector.
         */
        double
        l2_norm() const
        {
          double    result;
          const int ierr = column->Norm2(&result);
          AssertThrow(ierr == 0, ExcTrilinosError(ierr));

          return result;
        }

        /** double pointer to the first element of *this */
        const double *
        begin() const
        {
          double *values;
          column->ExtractView(&values);
          return values;
        }

        /** double pointer to the first element of *this */
        double *
        begin()
        {
          double *values;
          column->ExtractView(&values);
          return values;
        }

      private:
        std::unique_ptr<Epetra_Vector>
             column; /** Pointer to the matrix column */
        bool view;
      };



      /**
       * Default constructor.
       */
      MultiVector2()
        : Subscriptor()
        , vector(new Epetra_FEVector(
            Epetra_Map(0, 0, 0, Utilities::Trilinos::comm_self())))
      {}

      /**
       * Constructor.
       * @param v stores the rectangular matrix
       */
      MultiVector2(const Epetra_FEVector &v)
        : Subscriptor()
      {
        vector = std::make_unique<Epetra_FEVector>(v);
      }


      /**
       * Constructor.
       * @param v stores a rectangular matrix
       * @param n_vectors number of columns that we want to store in *this
       */
      MultiVector2(const Epetra_MultiVector &v, const int n_vectors)
        : Subscriptor()
      {
        int                 lda = v.Stride();
        std::vector<double> v_view(lda * v.NumVectors());
        v.ExtractCopy(v_view.data(), lda);
        vector = std::make_unique<Epetra_FEVector>(
          Copy, v.Map(), v_view.data(), lda, n_vectors);
      }

      /**
       * Copy constructor.
       */
      MultiVector2(const MultiVector2 &v)
        : Subscriptor()
      {
        vector = std::make_unique<Epetra_FEVector>(*v.vector);
      }


      /**
       * Destructor
       */
      ~MultiVector2() = default;

      /**
       * Assignment operator
       */
      MultiVector2 &
      operator=(const MultiVector2 &v)
      {
        Assert(vector.get() != nullptr,
               ExcMessage("Vector is not constructed properly."));

        vector = std::make_unique<Epetra_FEVector>(*v.vector);

        return *this;
      }



      /**
       * Assignment operator
       */
      MultiVector2 &
      operator=(MultiVector2 &&v) noexcept
      {
        std::swap(vector, v.vector);
        return *this;
      }

      /**
       * Sets all entries to @ref a.
       */
      void
      operator=(double a)
      {
        vector->PutScalar(a);
      }

      /** write Vector<double> to Preconditioners::MultiVector2 */
      MultiVector2 &
      operator=(const Vector<double> &v)
      {
        Assert(size() == v.size(), ExcDimensionMismatch(size(), v.size()));
        Assert(n_vectors() == 1,
               ExcMessage("This operator is only implemented for one column."));

        // this is probably not very efficient but works. in particular,  we
        // could do better if we know that number==TrilinosScalar because then
        // we could elide the copying of elements
        //
        // let's hope this isn't a particularly frequent operation
        for (int i = 0; i < vector->MyLength(); ++i)
          (*vector)[0][i] = v(i);

        return *this;
      }

      /**
       * Returns the underlying Epetra_FEVector that stores the rectangular
       * matrix.
       */
      Epetra_FEVector &
      get_columns() const
      {
        return *vector;
      }

      void
      clear()
      {
        // When we clear the vector, reset the pointer and generate an empty
        // vector.
#ifdef DEAL_II_WITH_MPI
        Epetra_Map map(0, 0, Epetra_MpiComm(MPI_COMM_SELF));
#else
        Epetra_Map map(0, 0, Epetra_SerialComm());
#endif

        // has_ghosts  = false;
        vector = std::make_unique<Epetra_FEVector>(map);
        // last_action = Zero;
      }


      /**
       * Reinits the object to a new matrix.
       */
      void
      reinit(const MultiVector2 &new_columns)
      {
        vector = std::make_unique<Epetra_FEVector>(new_columns.get_columns());
      }


      /**
       * Reinits the object to a new matrix.
       */
      void
      reinit(const Epetra_FEVector &new_columns)
      {
        vector = std::make_unique<Epetra_FEVector>(new_columns);
      }

      /**
       * Reinits the object to the first @ref n_columns columns of a new matrix @ref new_columns.
       */
      void
      reinit(const Epetra_MultiVector &new_columns, const int n_columns)
      {
        vector =
          std::make_unique<Epetra_FEVector>(new_columns.Map(), n_columns);
        IndexSet index_set(complete_index_set(new_columns.MyLength()));
        std::vector<unsigned int> indices(new_columns.MyLength());
        index_set.fill_index_vector(indices);
        const unsigned int length = new_columns.MyLength();
        for (int i = 0; i < n_columns; ++i)
          vector->ReplaceGlobalValues(length,
                                      reinterpret_cast<int *>(indices.data()),
                                      new_columns[i],
                                      i);
      }

      /**
       * Reinits the object to the first @ref n_columns columns of a new matrix @ref new_columns.
       */
      void
      reinit(const Preconditioners::MultiVector2 &new_columns,
             const int                            n_columns)
      {
        vector =
          std::make_unique<Epetra_FEVector>(new_columns.trilinos_vector().Map(),
                                            n_columns);
        IndexSet index_set(
          complete_index_set(new_columns.trilinos_vector().MyLength()));
        std::vector<unsigned int> indices(
          new_columns.trilinos_vector().MyLength());
        index_set.fill_index_vector(indices);
        const unsigned int length = new_columns.trilinos_vector().MyLength();
        for (int i = 0; i < n_columns; ++i)
          vector->ReplaceGlobalValues(length,
                                      reinterpret_cast<int *>(indices.data()),
                                      new_columns.trilinos_vector()[i],
                                      i);
      }

      /**
       * Reinits the matrix to size @ref length x @ref n_vectors.
       */
      void
      reinit(const int n_vectors, const int length)
      {
        Assert(
          n_vectors > 0,
          ExcMessage(
            "Initializing a MultiVector2 with zero columns is not possible."))
          Epetra_BlockMap map(length, 1, 0, Utilities::Trilinos::comm_world());
        vector = std::make_unique<Epetra_FEVector>(map, n_vectors);
      }



      /**
       * Reinits the object to a new matrix.
       */
      void
      reinit(const std::vector<VectorType> &new_columns)
      {
        const int number_of_columns = new_columns.size();
        const int length_of_columns = new_columns[0].size();
        vector                      = std::make_unique<Epetra_FEVector>(
          new_columns[0].trilinos_partitioner(), number_of_columns);
        IndexSet index_set(complete_index_set(length_of_columns));
        std::vector<unsigned int> indices(length_of_columns);
        index_set.fill_index_vector(indices);
        for (int i = 0; i < number_of_columns; ++i)
          vector->ReplaceGlobalValues(length_of_columns,
                                      reinterpret_cast<int *>(indices.data()),
                                      new_columns[i].begin(),
                                      i);
      }


      /**
       * Resize the matrix to a matrix with @ref n_cols columns with the size of @ref v.
       */
      void
      resize(const int n_cols, const TrilinosWrappers::MPI::Vector &v)
      {
        vector.reset();
        vector =
          std::make_unique<Epetra_FEVector>(v.trilinos_vector().Map(), n_cols);
      }

      /**
       * Number of columns.
       */
      unsigned int
      m() const
      {
        return vector->NumVectors();
      }

      /**
       * Length of columns.
       */
      unsigned int
      n() const
      {
        return vector->GlobalLength();
      }

      // needed for compability with anasazi solver
      unsigned int
      n_vectors() const
      {
        return vector->NumVectors();
      }

      // needed for compability with anasazi solver
      unsigned int
      size() const
      {
        return vector->GlobalLength();
      }

      /**
       * Fill the matrix with random values.
       */
      void
      fill_random()
      {
        std::mt19937                     gen((std::random_device())());
        const double                     mean    = 0.0;
        const double                     std_dev = 1.0; // standard deviation
        std::normal_distribution<double> nd(mean, std_dev);

        double *  entries;
        int       my_lda = vector->Stride();
        const int ierr   = vector->ExtractView(&entries, &my_lda);
        AssertThrow(ierr == 0, ExcTrilinosError(ierr));
        for (int i = 0; i < vector->NumVectors(); ++i)
          {
            for (int j = 0; j < vector->MyLength(); ++j)
              entries[i * my_lda + j] = nd(gen);
          }
      }

      /**
       * Scale the columns with the values contained in @ref scaling.
       */
      void
      scale(const TrilinosWrappers::MPI::Vector &scaling)
      {
        const int ierr =
          vector->Multiply(1.0, scaling.trilinos_vector(), *vector, 0.0);
        AssertThrow(ierr == 0, ExcTrilinosError(ierr));
      }

      /** Modified Gram-Schmidt, Alg. 1.2 in Iterative Methods for Sparse
       * Linear Systems*/
      void
      orthogonalize()
      {
        double column_norm;
        int    ierr = 0;
        (void)ierr;

        ierr = (*vector)(0)->Norm2(&column_norm);
        Assert(ierr == 0, ExcTrilinosError(ierr));
        Assert(column_norm != 0, ExcDivideByZero());

        column_norm = 1.0 / column_norm;
        ierr        = (*vector)(0)->Scale(column_norm);
        Assert(ierr == 0, ExcTrilinosError(ierr));

        for (int col = 1; col < (int)this->n_vectors(); ++col)
          {
            double start_value;
            ierr = (*vector)(col)->Norm2(&start_value);
            Assert(ierr == 0, ExcTrilinosError(ierr));
            for (int i = 0; i < col; ++i)
              {
                ierr = (*vector)(col)->Dot(*((*vector)(i)), &column_norm);
                Assert(ierr == 0, ExcTrilinosError(ierr));
                column_norm *= -1.0;
                ierr =
                  (*vector)(col)->Update(column_norm, *((*vector)(i)), 1.0);
                Assert(ierr == 0, ExcTrilinosError(ierr));
              }
            ierr = (*vector)(col)->Norm2(&column_norm);
            Assert(ierr == 0, ExcTrilinosError(ierr));

            // Check whether we want to reorthogonalize
            if (column_norm >
                10. * start_value *
                  std::sqrt(std::numeric_limits<double>::epsilon()))
              { // Assert(value != 0, ExcDivideByZero());
                column_norm = 1.0 / column_norm;
                ierr        = (*vector)(col)->Scale(column_norm);
                Assert(ierr == 0, ExcTrilinosError(ierr));
              }
            else // reorthogonalize
              {
                // reorthogonalize
                std::cout << "Reorthogonalize vectors." << std::endl;

                for (int i = 0; i < col; ++i)
                  {
                    // double value;
                    ierr = (*vector)(col)->Dot(*((*vector)(i)), &column_norm);
                    Assert(ierr == 0, ExcTrilinosError(ierr));
                    column_norm *= -1.0;
                    ierr =
                      (*vector)(col)->Update(column_norm, *((*vector)(i)), 1.0);
                    Assert(ierr == 0, ExcTrilinosError(ierr));
                  }
              }
            ierr = (*vector)(col)->Norm2(&column_norm);
            Assert(ierr == 0, ExcTrilinosError(ierr));
            column_norm = 1.0 / column_norm;
            ierr        = (*vector)(col)->Scale(column_norm);
            Assert(ierr == 0, ExcTrilinosError(ierr));
          }
      }

      /**
       * Access the first element of the @ref idx-th column.
       */
      Epetra_Vector &
      operator[](const int idx)
      {
        return *(*vector)(idx);
      }

      /**
       * Access the element @ref idx of the column.
       */
      const Epetra_Vector &
      operator[](const int idx) const
      {
        return *(*vector)(idx);
      }

      /** compute *this += src */
      void
      operator+=(const MultiVector2 &src)
      {
        const int ierr = vector->Update(1.0, *(src.vector), 1.);
        AssertThrow(ierr == 0, ExcTrilinosError(ierr));
      }

      /** Computes *this += factor * src. */
      void
      add(double factor, const MultiVector2 &src)
      {
        // TODO: assert dimensions
        AssertIsFinite(factor);

        const int ierr = vector->Update(factor, *(src.vector), 1.);
        AssertThrow(ierr == 0, ExcTrilinosError(ierr));
      }

      /**
       * Computes *this += factor1 * v1 + factor2 * v2
       */
      void
      add(double              factor1,
          const MultiVector2 &v1,
          double              factor2,
          const MultiVector2 &v2)
      {
        AssertIsFinite(factor1);
        AssertIsFinite(factor2);

        const int ierr =
          vector->Update(factor1, *(v1.vector), factor2, *(v2.vector), 1.);

        AssertThrow(ierr == 0, ExcTrilinosError(ierr));
      }

      /**
       * Computes 1.) *this *= factor1; 2.) *this += factor2 * v2;
       */
      void
      sadd(const double factor1, const double factor2, const MultiVector2 &v2)
      {
        AssertIsFinite(factor1);
        AssertIsFinite(factor2);

        const int ierr = vector->Update(factor2, *(v2.vector), factor1);

        AssertThrow(ierr == 0, ExcTrilinosError(ierr));
      }

      /**
       * scaled addition of the i-th and j-th column
       */
      void
      sadd(const double factor_i,
           const double factor_j,
           const int    i,
           const int    j)
      {
        AssertIsFinite(factor_i);
        AssertIsFinite(factor_j);
        Assert(i < (int)this->n_vectors(),
               ExcIndexRange(i, 0, this->n_vectors()));
        Assert(j < (int)this->n_vectors(),
               ExcIndexRange(j, 0, this->n_vectors()));

        const int ierr =
          (*vector)(i)->Update(factor_j, *((*vector)(j)), factor_i);
        (void)ierr;
        Assert(ierr == 0, ExcTrilinosError(ierr));
      }

      /** Multiplication with a scalar, *this *= factor */
      MultiVector2 &
      operator*=(const double factor)
      {
        AssertIsFinite(factor);

        const int ierr = vector->Scale(factor);
        AssertThrow(ierr == 0, ExcTrilinosError(ierr));

        return *this;
      }

      /** Multiply the i-th column with a scalar, *this *= factor */
      MultiVector2 &
      scale(const double factor, const int i)
      {
        AssertIsFinite(factor);

        const int ierr = (*vector)(i)->Scale(factor);
        AssertThrow(ierr == 0, ExcTrilinosError(ierr));

        return *this;
      }

      /** returns the dot product of the i-th and j-th column in @param result */
      void
      dot(const int i, const int j, double *result) const
      {
        (*vector)(i)->Dot(*((*vector)(j)), result);
      }

      /** Sets all values in the row @ref row to @ref value. */
      void
      set_row(const double value, const int row)
      {
        for (unsigned int i = 0; i < this->n_vectors(); ++i)
          {
            vector->ReplaceGlobalValues(1, &row, &value, i);
          }
      }

      /** Returns the l2-norm of the @ref i -th column of *this. */
      double
      l2_norm(const int i) const
      {
        double norm;
        (*vector)(i)->Norm2(&norm);
        return norm;
      }


      /**
       * Returns the underlying Epetra_FEVector that stores the matrix
       * entries.
       */
      Epetra_FEVector &
      trilinos_vector()
      {
        return *vector;
      }

      /**
       * Returns the underlying Epetra_FEVector that stores the matrix
       * entries.
       */
      // const Epetra_MultiVector &
      const Epetra_FEVector &
      trilinos_vector() const
      {
        // return static_cast<const Epetra_MultiVector &>(*vector);
        return *vector;
      }

      /**
       * Replaces the @ref i -th column with @ref src.
       */
      void
      set(const int i, TrilinosWrappers::MPI::Vector &src)
      {
        std::vector<unsigned int> indices(src.size());
        src.locally_owned_elements().fill_index_vector(indices);

        for (unsigned int j = 0; j < src.locally_owned_elements().size(); ++j)
          {
            const int row       = indices[j];
            const int local_row = vector->Map().LID(row);
            if (local_row != -1)
              (*vector)[i][local_row] = src[j];
            else
              {
                double    value = src[j];
                const int ierr =
                  vector->ReplaceGlobalValues(1, &row, &value, i);
                AssertThrow(ierr == 0, ExcTrilinosError(ierr));
              }
          }
      }

      /**
       * Returns the @ref i -th column of *this in @ref v.
       */
      void
      get(const int i, TrilinosWrappers::MPI::Vector &v) const
      {
        std::vector<unsigned int> indices(v.size());
        v.locally_owned_elements().fill_index_vector(indices);
        v.set(v.size(), indices.data(), (*vector)[i]);
      }


      /**
       * Computes the matrix-vector product
       * trg = (*this) *src.
       */
      void
      vmult(VectorType &trg, const VectorType &src) const
      {
        // Assert(src.size() == length_of_columns,
        //        ExcDimensionMismatch(src.size(), number_of_columns));
        // Assert(trg.size() == number_of_columns,
        //        ExcDimensionMismatch(trg.size(), length_of_columns));
        int ierr = trg.trilinos_vector().Multiply(
          'N', 'N', 1.0, *vector, src.trilinos_vector(), 0.0);
        AssertThrow(ierr == 0, ExcTrilinosError(ierr));
      }

      /**
       * Computes the transposed matrix-vector product
       * trg = (*this)^T *src.
       */
      void
      Tvmult(VectorType &trg, const VectorType &src) const
      {
        // Assert(src.size() == number_of_columns,
        //        ExcDimensionMismatch(trg.size(), number_of_columns));
        // Assert(trg.size() == length_of_columns,
        //        ExcDimensionMismatch(src.size(), length_of_columns));
        int ierr = trg.trilinos_vector().Multiply(
          'T', 'N', 1.0, *vector, src.trilinos_vector(), 0.0);
        AssertThrow(ierr == 0, ExcTrilinosError(ierr));
      }

      /**
       * Computes the transposed matrix-vector product
       * trg += (*this)^T *src.
       */
      void
      Tvmult_add(VectorType &trg, const VectorType &src) const
      {
        // Assert(src.size() == number_of_columns,
        //        ExcDimensionMismatch(trg.size(), number_of_columns));
        // Assert(trg.size() == length_of_columns,
        //        ExcDimensionMismatch(src.size(), length_of_columns));
        int ierr = trg.trilinos_vector().Multiply(
          'T', 'N', 1.0, *vector, src.trilinos_vector(), 1.0);
        AssertThrow(ierr == 0, ExcTrilinosError(ierr));
      }

      /**
       * Computes the matrix-vector product
       * trg = (*this) *src.
       */
      void
      vmult(Column &trg, const Column &src) const
      {
        // Assert(src.size() == length_of_columns,
        //        ExcDimensionMismatch(src.size(), number_of_columns));
        // Assert(trg.size() == number_of_columns,
        //        ExcDimensionMismatch(trg.size(), length_of_columns));
        int ierr = trg.trilinos_vector().Multiply(
          'N', 'N', 1.0, *vector, src.trilinos_vector(), 0.0);
        AssertThrow(ierr == 0, ExcTrilinosError(ierr));
      }

      /**
       * Computes the transposed matrix-vector product
       * trg = (*this)^T *src.
       */
      void
      Tvmult(Column &trg, const Column &src) const
      {
        // Assert(src.size() == number_of_columns,
        //        ExcDimensionMismatch(trg.size(), number_of_columns));
        // Assert(trg.size() == length_of_columns,
        //        ExcDimensionMismatch(src.size(), length_of_columns));
        int ierr = trg.trilinos_vector().Multiply(
          'T', 'N', 1.0, *vector, src.trilinos_vector(), 0.0);
        AssertThrow(ierr == 0, ExcTrilinosError(ierr));
      }


      /**
       * Computes the transposed matrix-vector product
       * trg += (*this)^T *src.
       */
      void
      Tvmult_add(Column &trg, const Column &src) const
      {
        // Assert(src.size() == number_of_columns,
        //        ExcDimensionMismatch(trg.size(), number_of_columns));
        // Assert(trg.size() == length_of_columns,
        //        ExcDimensionMismatch(src.size(), length_of_columns));
        int ierr = trg.trilinos_vector().Multiply(
          'T', 'N', 1.0, *vector, src.trilinos_vector(), 1.0);
        AssertThrow(ierr == 0, ExcTrilinosError(ierr));
      }



      /**
       * Matrix-matrix multiplication: Computes trg = (*this) * src
       */
      void
      mmult(MultiVector2 &trg, const MultiVector2 &src) const
      {
        // TODO Check dimensions:
        // Check length of column vectors
        // Assert(length_of_columns == src2.size(),
        //        ExcDimensionMismatch(length_of_columns, src2.size()));
        // // Check size of output matrix
        // Assert(number_of_columns == src1.size(),
        //        ExcDimensionMismatch(number_of_columns, src1.size()));
        // Assert(src2.n_vectors() == src1.n_vectors(),
        //        ExcDimensionMismatch(src2.n_vectors(), src1.n_vectors()));

        // perform dot product for all matrix entries:
        int ierr = trg.get_columns().Multiply(
          'N', 'N', 1.0, *vector, src.get_columns(), 0.0);
        AssertThrow(ierr == 0, ExcTrilinosError(ierr));
      }

      /**
       * Transposed matrix-matrix multiplication: Computes trg = (*this)^T *
       * src
       */
      void
      Tmmult(MultiVector2 &trg, const MultiVector2 &src) const
      {
        AssertDimension(vector->GlobalLength(), src.size());
        AssertDimension(trg.size(), vector->NumVectors());
        AssertDimension(trg.n_vectors(), src.n_vectors());

        int ierr = trg.get_columns().Multiply(
          'T', 'N', 1.0, *vector, src.get_columns(), 0.0);
        AssertThrow(ierr == 0, ExcTrilinosError(ierr));
      }

      /**
       * Matrix-matrix multiplication: Computes trg = (*this) * src
       */
      void
      mmult_add(MultiVector2 &trg, const MultiVector2 &src) const
      {
        // TODO Check dimensions:
        // Check length of column vectors
        // Assert(length_of_columns == src2.size(),
        //        ExcDimensionMismatch(length_of_columns, src2.size()));
        // // Check size of output matrix
        // Assert(number_of_columns == src1.size(),
        //        ExcDimensionMismatch(number_of_columns, src1.size()));
        // Assert(src2.n_vectors() == src1.n_vectors(),
        //        ExcDimensionMismatch(src2.n_vectors(), src1.n_vectors()));

        // perform dot product for all matrix entries:
        int ierr = trg.get_columns().Multiply(
          'N', 'N', 1.0, *vector, src.get_columns(), 1.0);
        AssertThrow(ierr == 0, ExcTrilinosError(ierr));
      }

      /**
       * Transposed matrix-matrix multiplication: Computes trg = (*this)^T *
       * src
       */
      void
      Tmmult_add(MultiVector2 &trg, const MultiVector2 &src) const
      {
        AssertDimension(vector->GlobalLength(), src.size());
        AssertDimension(trg.size(), vector->NumVectors());
        AssertDimension(trg.n_vectors(), src.n_vectors());

        int ierr = trg.get_columns().Multiply(
          'T', 'N', 1.0, *vector, src.get_columns(), 1.0);
        AssertThrow(ierr == 0, ExcTrilinosError(ierr));
      }



      /**
       * Scale the columns with the entries of @ref scaling.
       */
      void
      scale(const std::vector<double> &scaling)
      {
        unsigned int number_of_columns = vector->NumVectors();

        Assert(scaling.size() == number_of_columns,
               ExcDimensionMismatch(scaling.size(), number_of_columns));


        for (unsigned int i = 0; i < number_of_columns; ++i)
          {
            const unsigned int ierr = (*vector)(i)->Scale(scaling[i]);
            AssertThrow(ierr == 0, ExcTrilinosError(ierr));
          }
      }

      /**
       * Points the first entry of the first column.
       */
      double *
      begin()
      {
        return (*vector)[0];
      }

      /**
       * Points to the element past the last entry of the last column.
       */
      double *
      end()
      {
        return (*vector)[0] + local_size();
      }

      /**
       * Points the first entry of the first column.
       */
      double *
      begin() const
      {
        return (*vector)[0];
      }

      /**
       * Points to the element past the last entry of the last column.
       */
      double *
      end() const
      {
        return (*vector)[0] + local_size();
      }

      /**
       * Returns the number of matrix entries.
       */
      int
      local_size() const
      {
        return vector->MyLength() * vector->NumVectors();
      }

      /** Prints the matrix. */
      void
      print(std::ostream &out) const
      {
        vector->Print(out);
      }



    private:
      std::unique_ptr<Epetra_FEVector> vector;
    };
  } // namespace Preconditioners


  template class VectorMemory<Preconditioners::MultiVector2::Column>;
  template class GrowingVectorMemory<Preconditioners::MultiVector2::Column>;
} // namespace dealii


#endif
