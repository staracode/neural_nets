////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//																			  //
//	  Neural Network Program with binary (1.0 and 0.0) inputs and outputs.	  //
//																			  //
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#define MAX_TESTS (10000)

//#define NN_8x3x8			// comment or un-comment

#define NN_rap1				// keep always defined - NEVER COMMENT OUT

//#define NN_rap1_binary_encoding	// comment or un-comment

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#include <assert.h>
#include <ctime>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

using namespace std;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

fstream logFile;

// random generator function:
ptrdiff_t arandom (ptrdiff_t i) { return rand() % i; }

// pointer object to it:
ptrdiff_t (*arandomp)(ptrdiff_t) = arandom;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

class Vertex
{
	int		size;
	double *node;
	double *grad;
	Vertex(){}
public:
	Vertex( int size_ ) : size(size_)
	{
		node = new double[ size_ ];
		grad = new double[ size_ ];

		for ( int n = 0; n < size_; n++ )
			node[ n ] = 0.0;
	}
	double& operator[]( const int n_ )
	{
		assert( n_ >= 0 && n_ < size );

		return node[ n_ ];
	}
	double& operator()( const int n_ )
	{
		assert( n_ >= 0 && n_ < size );

		return grad[ n_ ];
	}
};

////////////////////////////////////////////////////////////////////////////////

class Edge
{
	int	   in, out;
	double **edge;
	Edge(){}
public:
	Edge( int in_, int out_ ) : in(in_),  out(out_)
	{
		edge = new double*[ in_ ];
		for( int i = 0 ; i < in_ ; i++ )
			edge[ i ] = new double[ out_ ];
	}
	double& operator()( const int in_, const int out_ )
	{
		assert( in_  >= 0 && in_  < in  );
		assert( out_ >= 0 && out_ < out );

		return edge[ in_ ][ out_ ];
	}
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

class NN
{
	int inputs, hiddens, outputs, isymbols, osymbols,
		input_nodes_per_input_symbol, output_nodes_per_output_symbol;

	Vertex *iNp, *hNp, *oNp;
	Edge   *ihWeightp, *hoWeightp, *ihDelp, *hoDelp;

	int	   seed;
	double rate;
	double max_error;
	int	   max_epochs;
	int	   min_epoch_size;
	int	   max_epoch_size;

	int	   positives, negatives;
	float  **pos_inputs, **pos_outputs, **neg_inputs, **neg_outputs;

	map< char, float* > isymbmap, osymbmap;

	//-----------------------------------------------------
	//-----------------------------------------------------
	//-----------------------------------------------------

	void FF( float *inputs_ )
	{
		for ( int i = 0; i < inputs; i++ )
			(*iNp)[ i ] = inputs_[ i ];

		for ( int i = 0; i < hiddens; i++ )
		{
			(*hNp)[ i ] = 0.0;

			for ( int j = 0; j <= inputs; j++ )
				(*hNp)[ i ] += (*iNp)[ j ] * (*ihWeightp)( j, i );

			(*hNp)[ i ] = 1.0 / ( 1.0 + exp( - (*hNp)[ i ] ) );
		}

		for ( int i = 0; i < outputs; i++ )
		{
			(*oNp)[ i ] = 0.0;

			for ( int j = 0; j <= hiddens; j++ )
				(*oNp)[ i ] += (*hNp)[ j ] * (*hoWeightp)( j, i );

			(*oNp)[ i ] = 1.0 / ( 1.0 + exp( - (*oNp)[ i ] ) );
		}
	}

	//-----------------------------------------------------

	void BP( float *outputs_ )
	{
		for ( int i = 0; i < outputs; i++ )
		{
			double out = (*oNp)[ i ];
			(*oNp)( i ) = out * ( 1.0 - out ) * ( outputs_[ i ] - out );

			for ( int j = 0; j <= hiddens; j++ )
				(*hoDelp)( j, i ) += rate * (*hNp)[ j ] * (*oNp)( i );
		}

		for ( int i = 0; i < hiddens; i++ )
		{
			double sum = 0.0;
			for( int k = 0; k < outputs; k++ )
				sum += (*hoWeightp)( i, k ) * (*oNp)( k );

			double hid = (*hNp)[ i ];
			(*hNp)( i ) = hid * ( 1.0 - hid ) * sum;

			for ( int j = 0; j <= inputs; j++ )
				(*ihDelp)( j, i ) += rate * (*iNp)[ j ] * (*hNp)( i );
		}
		//update();
	}

	//-----------------------------------------------------

	void update()
	{
		for ( int i = 0; i <= inputs; i++ )
			for ( int j = 0; j < hiddens; j++ )
			{
				(*ihWeightp)( i, j ) += (*ihDelp)( i, j );
 				(*ihDelp)( i, j ) = 0.0;
			}

		for ( int i = 0; i <= hiddens; i++ )
			for ( int j = 0; j < outputs; j++ )
			{
				(*hoWeightp)( i, j ) += (*hoDelp)( i, j );
				(*hoDelp)( i, j ) = 0.0;
			}
	}

	//-----------------------------------------------------
	//-----------------------------------------------------
	//-----------------------------------------------------

	void mapsymbol2float( map< char, float* > & symbolmap_, int nodes_per_symbol_ )
	{
		int symbols = symbolmap_.size();
		map< char, float* >::iterator symbolmapit = symbolmap_.begin();

		if ( symbols == nodes_per_symbol_ )
			for ( int i = 0; i < symbols; i++, symbolmapit++ )
			{
				symbolmapit->second = new float[ nodes_per_symbol_ ];

				for ( int j = 0; j < nodes_per_symbol_; j++ )
					(symbolmapit->second)[ j ] = (float)( i == j ? 1.0 : 0.0 );					// one-hot encoding
				//  (symbolmapit->second)[ j ] = (float)( ( ( ( 1 << i ) >> j ) & 1 ) * 1.0 );	// one-hot encoding - same as above
			}
		else
			for ( int i = 0; i < symbols; i++, symbolmapit++ )
			{
				symbolmapit->second = new float[ nodes_per_symbol_ ];

				for ( int j = 0; j < nodes_per_symbol_; j++ )
					(symbolmapit->second)[ j ] = (float)( ( ( ( i ) >> j ) & 1 ) * 1.0 );	// binary encoding
			}
	}

	//-----------------------------------------------------

	void logsymbol2float( map< char, float* > & symbolmap_, int nodes_per_symbol_ )
	{
		int symbols = symbolmap_.size();
		map< char, float* >::iterator symbolmapit = symbolmap_.begin();

		for ( int i = 0; i < symbols; i++, symbolmapit++ )
		{
			logFile << symbolmapit->first << " =  ";
			for ( int j = 0; j < nodes_per_symbol_-1; j++ )
				logFile << (symbolmapit->second)[ j ] << ", ";
			logFile << (symbolmapit->second)[ nodes_per_symbol_-1 ] << "\n";
		}
	}

	//-----------------------------------------------------
	//-----------------------------------------------------

	void readTrainingVectors( const char *posFileName, const char *negFileName )
	{
		string line = "";

 		int input_symbols  = inputs  / input_nodes_per_input_symbol;
#if   defined NN_8x3x8
 		int output_symbols = outputs / output_nodes_per_output_symbol;
		int line_length = input_symbols + output_symbols;
#elif defined NN_rap1
		int line_length = input_symbols;
#endif

		vector< string > *posLines = new vector< string >;
		vector< string > *negLines = new vector< string >;

		fstream posFile;
		posFile.open( posFileName, ios::in );
		positives = 0;
		while ( !posFile.eof() )
		{
			getline( posFile, line );

			if( line.length() == (unsigned)0 ) continue;
			assert( line.length() == (unsigned)line_length );

			for ( int i = 0; i < input_symbols; i++ )
				isymbmap[ line[ i ] ];
#if   defined NN_8x3x8
			for ( int i = input_symbols; i < line_length; i++ )
				osymbmap[ line[ i ] ];
#elif defined NN_rap1
			osymbmap[ 'p' ];
#endif

			posLines->push_back( line );
			positives++;
		}
		posFile.close();

		fstream negFile;
		negFile.open( negFileName, ios::in );
		negatives = 0;
		while ( !negFile.eof() )
		{
			getline( negFile, line );

			if( line.length() == (unsigned)0 ) continue;
			assert( line.length() == (unsigned)line_length );

			for ( int i = 0; i < input_symbols; i++ )
				isymbmap[ line[ i ] ];
#if   defined NN_8x3x8
			for ( int i = input_symbols; i < line_length; i++ )
				osymbmap[ line[ i ] ];
#elif defined NN_rap1
			osymbmap[ 'n' ];
#endif

			negLines->push_back( line );
			negatives++;
		}
		negFile.close();

		assert( isymbmap.size() == (unsigned)isymbols );
		assert( osymbmap.size() == (unsigned)osymbols );

		mapsymbol2float( isymbmap, input_nodes_per_input_symbol   );
		mapsymbol2float( osymbmap, output_nodes_per_output_symbol );

		logFile << "input symbol and input nodes:\n";
		logsymbol2float( isymbmap, input_nodes_per_input_symbol   );
		logFile << "output symbol and output nodes:\n";
		logsymbol2float( osymbmap, output_nodes_per_output_symbol );
		logFile << "\n";

		random_shuffle( posLines->begin(), posLines->end(), arandomp );
		random_shuffle( negLines->begin(), negLines->end(), arandomp );

		vector< string >::iterator linesit;
		string::iterator symbit;

		pos_inputs = new float*[ positives ];
		for( int i = 0 ; i < positives ; i++ )
			pos_inputs[ i ] = new float[ inputs ];
		pos_outputs = new float*[ positives ];
		for( int i = 0 ; i < positives ; i++ )
			pos_outputs[ i ] = new float[ outputs ];

		linesit = posLines->begin();
		for ( int i = 0; i < positives; i++, linesit++ )
		{
			symbit = linesit->begin();
			for ( int j = 0; j < input_symbols; j++, symbit++ )
				for ( int k = 0; k < input_nodes_per_input_symbol; k++ )
					pos_inputs[ i ][ j * input_nodes_per_input_symbol + k ] = isymbmap[ *symbit ][ k ];
#if   defined NN_8x3x8
			for ( int j = 0; j < output_symbols; j++, symbit++ )
				for ( int k = 0; k < output_nodes_per_output_symbol; k++ )
					pos_outputs[ i ][ j * output_nodes_per_output_symbol + k ] = osymbmap[ *symbit ][ k ];
#elif defined NN_rap1
			pos_outputs[ i ][ 0 ] = osymbmap[ 'p' ][ 0 ];
#endif
		}

		neg_inputs = new float*[ negatives ];
		for( int i = 0 ; i < negatives ; i++ )
			neg_inputs[ i ] = new float[ inputs ];
		neg_outputs = new float*[ negatives ];
		for( int i = 0 ; i < negatives ; i++ )
			neg_outputs[ i ] = new float[ outputs ];

		linesit = negLines->begin();
		for ( int i = 0; i < negatives; i++, linesit++ )
		{
			symbit = linesit->begin();
			for ( int j = 0; j < input_symbols; j++, symbit++ )
				for ( int k = 0; k < input_nodes_per_input_symbol; k++ )
					neg_inputs[ i ][ j * input_nodes_per_input_symbol + k ] = isymbmap[ *symbit ][ k ];
#if   defined NN_8x3x8
			for ( int j = 0; j < output_symbols; j++, symbit++ )
				for ( int k = 0; k < output_nodes_per_output_symbol; k++ )
					neg_outputs[ i ][ j * output_nodes_per_output_symbol + k ] = osymbmap[ *symbit ][ k ];
#elif defined NN_rap1
			neg_outputs[ i ][ 0 ] = osymbmap[ 'n' ][ 0 ];
#endif
		}

		posLines->clear();
		negLines->clear();
		delete posLines;
		delete negLines;
	}

	//-----------------------------------------------------
	//-----------------------------------------------------

	void displayN()
	{
		double value;

		for ( int i = 0; i < inputs+1; i++ )
		{
			value = (*iNp)[ i ];
			cout << "input node: (*iNp)( " << i << " ) = " << value << '\n';
		}
		cout << '\n';

		for ( int i = 0; i < hiddens+1; i++ )
		{
			value = (*hNp)[ i ];
			cout << "hidden node: (*hNp)( " << i << " ) = " << value << '\n';
		}
		cout << '\n';

		for ( int i = 0; i < outputs; i++ )
		{
			value = (*oNp)[ i ];
			cout << "output node: (*oNp)( " << i << " ) = " << value << '\n';
		}
		cout << '\n';
	}

	//-----------------------------------------------------

	void displayW()
	{
		double value;

		for ( int i = 0; i < inputs+1; i++ )
			for ( int j = 0; j < hiddens; j++ )
			{
				value = (*ihWeightp)( i, j );
				cout << "input to hidden weight: (*ihWeightp)( " << i << ", " << j << " ) = " << value << '\n';
			}
		cout << '\n';

		for ( int i = 0; i < hiddens+1; i++ )
			for ( int j = 0; j < outputs; j++ )
			{
				value = (*hoWeightp)( i, j );
				cout << "hidden to output weight: (*hoWeightp)( " << i << ", " << j << " ) = " << value << '\n';
			}
		cout << '\n';
	}

	//-----------------------------------------------------
	//-----------------------------------------------------
	//-----------------------------------------------------

public:
	NN( int inputs_, int hiddens_, int outputs_, int isymbols_, int osymbols_,
		int input_nodes_per_input_symbol_, int output_nodes_per_output_symbol_ )
	: inputs(inputs_), hiddens(hiddens_), outputs(outputs_), isymbols(isymbols_), osymbols(osymbols_),
	  input_nodes_per_input_symbol(input_nodes_per_input_symbol_),
	  output_nodes_per_output_symbol(output_nodes_per_output_symbol_)
	{
		assert( isymbols_ > 1 );
		assert( osymbols_ > 1 );
		assert( isymbols_ == input_nodes_per_input_symbol_   || isymbols_ == pow( 2.0, input_nodes_per_input_symbol_   ) );
		assert( osymbols_ == output_nodes_per_output_symbol_ || osymbols_ == pow( 2.0, output_nodes_per_output_symbol_ ) );

		iNp = new Vertex( inputs_ +1 );   (*iNp)[ inputs_  ] = -1.0;
		hNp = new Vertex( hiddens_+1 );   (*hNp)[ hiddens_ ] = -1.0;
		oNp = new Vertex( outputs_   );

		ihWeightp = new Edge( inputs_ +1, hiddens_ );
		hoWeightp = new Edge( hiddens_+1, outputs_ );

		ihDelp = new Edge( inputs_ +1, hiddens_ );
		hoDelp = new Edge( hiddens_+1, outputs_ );
	}

	//-----------------------------------------------------

	void init( int seed_, double rate_, double max_error_, int max_epochs_, int min_epoch_size_, int max_epoch_size_ )
	{
 		seed		   = seed_;
		rate		   = rate_;
		max_error	   = max_error_;
		max_epochs	   = max_epochs_;
		min_epoch_size = min_epoch_size_;
		max_epoch_size = max_epoch_size_;

 		srand( seed );
		for ( int i = 0; i <= inputs; i++ )
			for ( int j = 0; j < hiddens; j++ )
			{
				(*ihWeightp)( i, j ) = (double)arandom(RAND_MAX) / (RAND_MAX + 1) - 0.5;
				(*ihDelp)( i, j )	= 0.0;
			}
		for ( int i = 0; i <= hiddens; i++ )
			for ( int j = 0; j < outputs; j++ )
			{
				(*hoWeightp)( i, j ) = (double)arandom(RAND_MAX) / (RAND_MAX + 1) - 0.5;
				(*hoDelp)( i, j )	= 0.0;
			}
	}

	//-----------------------------------------------------

	void train( const char* posFileName, const char* negFileName )
	{
		readTrainingVectors( posFileName, negFileName );

		double err2, max_error2 = max_error * max_error;

		int epoch = 0, epoch_size = 0, posx = 0, negx = 0;

		while ( epoch < max_epochs )
		{
			err2 = 0.0;

			epoch_size = epoch_size + min_epoch_size;
			//epoch_size = min( epoch_size, max( positives, negatives ) );
			//epoch_size = min( epoch_size, min( positives, negatives ) );
			epoch_size = min( epoch_size, max_epoch_size );

			for ( int e = 0; e < epoch_size; e++ )
			{
				FF( pos_inputs [ posx ] );
				BP( pos_outputs[ posx ] );

				for ( int i = 0; i < outputs; i++ )
				{
					double err = (*oNp)[ i ] - pos_outputs[ posx ][ i ];
					err2 += err * err;
				}
				if ( ++posx == positives )   posx = 0;

				FF( neg_inputs [ negx ] );
				BP( neg_outputs[ negx ] );

				for ( int i = 0; i < outputs; i++ )
				{
					double err = (*oNp)[ i ] - neg_outputs[ negx ][ i ];
					err2 += err * err;
				}
				if ( ++negx == negatives )   negx = 0;
			}
			update();
			epoch++;

			err2 = err2 / ( outputs * epoch_size );

			cout	<< "epoch = " << epoch << "\terr^2 = " << err2 << "\n";
			logFile << "epoch = " << epoch << "\terr^2 = " << err2 << "\n";

			if ( err2 < max_error2 )  break;
		}

		cout	<< "training ended\n\n";
		logFile << "training ended\n\n";
	}

	//-----------------------------------------------------

	void test( const char *testFileName )
	{
 		int input_symbols  = inputs  / input_nodes_per_input_symbol;
 		int output_symbols = outputs / output_nodes_per_output_symbol;

		int line_length1 = input_symbols;
		int line_length2 = input_symbols + output_symbols;

		fstream testFile;
		testFile.open( testFileName, ios::in );

		string line = "";

		float *test_inputs  = new float[ inputs  ];
		float *test_outputs = new float[ outputs ];

		int  tests = 0;
 		int  vector_error_count = 0;
		bool compare_outputs = false;

		while ( !testFile.eof() && tests < MAX_TESTS )
		{
			getline( testFile, line );

			int line_length = line.length();
			if ( line_length == 0 ) continue;
			if ( line_length != line_length1 && line_length != line_length2 )
			{
				cout	<< "line length error for test vector = " << line << " (test = " << tests << ")\n";
				logFile << "line length error for test vector = " << line << " (test = " << tests << ")\n";
				return;
			}

			assert( line.length() == (unsigned)line_length );
			if ( line_length == line_length2 )
				compare_outputs = true;

			string::iterator symbit = line.begin();
			for ( int i = 0; i < input_symbols; i++, symbit++ )
				if ( !isymbmap.count( line[ i ] ) )
				{
					cout	<< "input symbol error for test vector = " << line << " (test = " << tests << ")\n";
					logFile << "input symbol error for test vector = " << line << " (test = " << tests << ")\n";
					return;
				}
				else
				{
					for ( int j = 0; j < input_nodes_per_input_symbol; j++ )
						test_inputs[ i * input_nodes_per_input_symbol + j ] = isymbmap[ *symbit ][ j ];
				}
			if ( compare_outputs )
				for ( int i = 0; i < output_symbols; i++, symbit++ )
					if ( !osymbmap.count( line[ input_symbols + i ] ) )
					{
						cout	<< "output symbol error for test vector = " << line << " (test = " << tests << ")\n";
						logFile << "output symbol error for test vector = " << line << " (test = " << tests << ")\n";
						return;
					}
					else
					{
						for ( int j = 0; j < output_nodes_per_output_symbol; j++ )
							test_outputs[ i * output_nodes_per_output_symbol + j ] = osymbmap[ *symbit ][ j ];
					}


			FF( test_inputs );

			double sum = 0.0;
			int bit_error_count = 0;
			for ( int i = 0; i < outputs; i++ )
			{
				sum += (*oNp)[ i ];

				if ( line_length == line_length2 )
				{
					int actual_output  = (int)( (*oNp)[ i ]	   + 0.5 );
					int desired_output = (int)( test_outputs[ i ] + 0.5 );
					if ( actual_output != desired_output )
						bit_error_count++;
				}
			}
			if ( compare_outputs )
				if ( bit_error_count )
					vector_error_count++;


			cout	<< line << "\t" << sum << "\t(test = " << tests << ")\tout = ";
			for ( int i = 0; i < outputs-1; i++ )
				cout	<< (*oNp)[ i ] << ", ";
			if ( compare_outputs )
				cout	<< (*oNp)[ outputs-1 ] << "\tbit_error_count = " << bit_error_count << "\n";
			else
				cout	<< (*oNp)[ outputs-1 ] << "\n";

			logFile << line << "\t" << sum << "\t(test = " << tests << ")\tout = ";
			for ( int i = 0; i < outputs-1; i++ )
				logFile << (*oNp)[ i ] << ", ";
			if ( compare_outputs )
				logFile << (*oNp)[ outputs-1 ] << "\tbit_error_count = " << bit_error_count << "\n";
			else
				logFile << (*oNp)[ outputs-1 ] << "\n";

			////logFile << line << "\t" << sum << "\n";

			tests++;
		}
		testFile.close();

		if ( compare_outputs )
		{
			cout	<< "vector_error_count = " << vector_error_count << "\n";
			logFile << "vector_error_count = " << vector_error_count << "\n";
		}

		delete [] test_inputs;
		delete [] test_outputs;

		cout	<< "testing stopped\n\n";
		logFile << "testing stopped\n\n";
	}

	//-----------------------------------------------------
	//-----------------------------------------------------

	void logW()
	{
		double value;

		logFile << "------------------------------------------------------------------------\n";
		for ( int i = 0; i < inputs+1; i++ )
			for ( int j = 0; j < hiddens; j++ )
			{
				value = (*ihWeightp)( i, j );
				logFile << "input to hidden weight: (*ihWeightp)( " << i << ", " << j << " ) = " << value << '\n';
			}
		logFile << "------------------------------------------------------------------------\n";
		for ( int i = 0; i < hiddens+1; i++ )
			for ( int j = 0; j < outputs; j++ )
			{
				value = (*hoWeightp)( i, j );
				logFile << "hidden to output weight: (*hoWeightp)( " << i << ", " << j << " ) = " << value << '\n';
			}
		logFile << "------------------------------------------------------------------------\n\n";
	}

	//-----------------------------------------------------

	void log_parameters()
	{
		logFile << "						 MAX_TESTS = " <<					   MAX_TESTS << '\n';

		logFile << "							inputs = " <<						  inputs << '\n';
		logFile << "						   hiddens = " <<						 hiddens << '\n';
		logFile << "						   outputs = " <<						 outputs << '\n';
		logFile << "						  isymbols = " <<						isymbols << '\n';
		logFile << "						  osymbols = " <<						osymbols << '\n';

		logFile << "	  input_nodes_per_input_symbol = " <<   input_nodes_per_input_symbol << '\n';
		logFile << "	output_nodes_per_output_symbol = " << output_nodes_per_output_symbol << '\n';

 		logFile << "							  seed = " <<							seed << '\n';
		logFile << "							  rate = " <<							rate << '\n';
		logFile << "						 max_error = " <<					   max_error << '\n';
		logFile << "						max_epochs = " <<					  max_epochs << '\n';
		logFile << "					min_epoch_size = " <<				  min_epoch_size << '\n';
		logFile << "					max_epoch_size = " <<				  max_epoch_size << '\n';
	}

	//-----------------------------------------------------
	//-----------------------------------------------------

	void roc( const char *rocFileName )
	{
		fstream rocFile;
		rocFile.open( rocFileName, ios::out );
		rocFile.setf( ios::fixed, ios::floatfield );
		rocFile.precision( 3 );

		float *posScores = new float[ positives ];
		float *negScores = new float[ positives ];

		for ( int j = 0; j < positives; j++ )
		{
			FF( pos_inputs [ j ] );
			posScores[ j ] = (float)(*oNp)[ 0 ];
			FF( neg_inputs [ j ] );
			negScores[ j ] = (float)(*oNp)[ 0 ];
		}

		float TPy[ 100 ];
		float FPx[ 100 ];

		for ( int i = 0; i < 100; i++ )
		{
			int TP = 0, FP = 0;

			for ( int j = 0; j < positives; j++ )
			{
				if ( posScores[ j ] > ((float)i) / 100 )  TP++;
				if ( negScores[ j ] > ((float)i) / 100 )  FP++;
			}
			TPy[ i ] = ((float)TP) / positives;
			FPx[ i ] = ((float)FP) / positives;
		}

		for ( int j = 0; j < 100-1; j++ ) rocFile << FPx[ j ] << ",";
		rocFile << FPx[ 100-1 ] << "\n";

		for ( int j = 0; j < 100-1; j++ ) rocFile << TPy[ j ] << ",";
		rocFile << TPy[ 100-1 ] << "\n";

		rocFile.close();

		cout	<< "roc completed\n\n";
		logFile << "roc completed\n\n";
	}

	//-----------------------------------------------------
	//-----------------------------------------------------
	//-----------------------------------------------------
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


#if   defined NN_8x3x8

#define LOG_FILENAME    "8x3x8.log.txt"
#define POS_FILENAME    "8x3x8.pos.txt"
#define NEG_FILENAME    "8x3x8.neg.txt"
#define TST_FILENAME    "8x3x8.test.txt"
#define ROC_FILENAME    "8x3x8.roc.csv"

#define INPUT_NODES_PER_INPUT_SYMBOL   (1)
#define OUTPUT_NODES_PER_OUTPUT_SYMBOL (1)

#define INPUT_NODES     (8 *INPUT_NODES_PER_INPUT_SYMBOL)
#define HIDDEN_NODES    (3)
#define OUTPUT_NODES    (8 *OUTPUT_NODES_PER_OUTPUT_SYMBOL)
#define INPUT_SYMBOLS   (2)
#define OUTPUT_SYMBOLS  (2)

//#define RANDOM_SEED     (547767)
#define RANDOM_SEED     (0)
#define LEARNING_RATE   (0.1)
#define MAX_ERROR       (0.01)
#define EPOCHS          (2000)
#define MIN_EPOCH_SIZE  (100)
#define MAX_EPOCH_SIZE  (280)

#elif defined NN_rap1

#define LOG_FILENAME    "rap1.log.txt"
#define POS_FILENAME    "rap1.pos.txt"
#define NEG_FILENAME    "rap1.neg.txt"
#define TST_FILENAME    "rap1.test.txt"
#define ROC_FILENAME    "rap1.roc.csv"

#define INPUT_NODES_PER_INPUT_SYMBOL	(4)	// one-hot encoding
#define OUTPUT_NODES_PER_OUTPUT_SYMBOL	(1)

#define INPUT_NODES     (17 *INPUT_NODES_PER_INPUT_SYMBOL)
#define HIDDEN_NODES    (7)
#define OUTPUT_NODES    (1  *OUTPUT_NODES_PER_OUTPUT_SYMBOL)
#define INPUT_SYMBOLS   (4)
#define OUTPUT_SYMBOLS  (2)

//#define RANDOM_SEED     (547767)
#define RANDOM_SEED     (0)
#define LEARNING_RATE   (0.1)
#define MAX_ERROR       (0.001)
#define EPOCHS          (100)
#define MIN_EPOCH_SIZE  (100)
#define MAX_EPOCH_SIZE  (500)

#endif

//--------------------------------------------------

#if  !defined NN_8x3x8
#if   defined NN_rap1
#if   defined NN_rap1_binary_encoding


#undef INPUT_NODES_PER_INPUT_SYMBOL

#undef INPUT_NODES
#undef HIDDEN_NODES

#undef RANDOM_SEED
#undef LEARNING_RATE
#undef MAX_ERROR
#undef EPOCHS
#undef MIN_EPOCH_SIZE
#undef MAX_EPOCH_SIZE


#define INPUT_NODES_PER_INPUT_SYMBOL	(2)	// binary encoding

#define INPUT_NODES     (17 *INPUT_NODES_PER_INPUT_SYMBOL)
#define HIDDEN_NODES    (8)

//#define RANDOM_SEED     (547767)
#define RANDOM_SEED     (0)
#define LEARNING_RATE   (0.1)
#define MAX_ERROR       (0.001)
#define EPOCHS          (1000)
#define MIN_EPOCH_SIZE  (100)
#define MAX_EPOCH_SIZE  (500)


#endif
#endif
#endif

////////////////////////////////////////////////////////////////////////////////

int main()
{
	cout.setf( ios::fixed, ios::floatfield );
	cout.precision( 5 );

	logFile.open( LOG_FILENAME, ios::out );
	logFile.setf( ios::fixed, ios::floatfield );
	logFile.precision( 5 );

	NN nn( INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES, INPUT_SYMBOLS, OUTPUT_SYMBOLS,
		   INPUT_NODES_PER_INPUT_SYMBOL, OUTPUT_NODES_PER_OUTPUT_SYMBOL );
	nn.init( RANDOM_SEED, LEARNING_RATE, MAX_ERROR, EPOCHS, MIN_EPOCH_SIZE, MAX_EPOCH_SIZE );

	logFile << "==================================================TRAINING===============\n\n";
	time_t start_time = time(NULL);
	logFile << "START TIME =  " << start_time << " secs\n\n";
	nn.train( POS_FILENAME, NEG_FILENAME );
	time_t end_time = time(NULL);
	logFile << "  END TIME =  " << end_time << " secs\n";
	time_t diff_time = difftime( end_time, start_time );
	logFile << " DIFF TIME =  " << diff_time << " secs\n\n";
	nn.logW();
	logFile << "=========================================================================\n\n";

	logFile << "==================================================HEGATIVES==============\n";
	nn.test( NEG_FILENAME );
	logFile << "==================================================POSITIVES==============\n";
	nn.test( POS_FILENAME );
	logFile << "==================================================TESTS==================\n";
	nn.test( TST_FILENAME );
	logFile << "==================================================PARAMETERS=============\n";
	nn.log_parameters();
	logFile << "=========================================================================\n\n";

#if  !defined NN_8x3x8
	nn.roc( ROC_FILENAME );
#endif

	cout	<< "PROGRAM ENDED\n";
	logFile << "PROGRAM ENDED\n";
	logFile.close();

	return 0;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
