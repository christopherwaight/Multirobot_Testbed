package edu.scu.engr.rsl.deca;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.Socket;
import java.net.UnknownHostException;
import java.nio.charset.Charset;
import java.util.Date;

public class SocketShim {

	Socket sock;
	InputStream readStream;
	OutputStream writeStream;

	boolean open = false;

	static void usageAndExit( ) {
		System.err.println( "Spoof the TCP-based robot controller.\n" +
			"Usage: java -jar SocketShim.jar address [drivecommand (x;y;z)] [port]\n" +
			"Example: SocketShim.jar 192.168.176.25 \"0.1;0;0.1\" 57028" );
		System.exit( 1 );
	}

	public static void main( String[] argv ) throws ShimException, UnknownHostException {

		String address = "localhost";
		int port = 0xDEC4;
		String spam = "0;0;0";

		switch ( argv.length ) {
			case 3:
				try {
					port = Integer.parseInt( argv[2] );
				} catch ( NumberFormatException e ) {
					System.err.println( "Port parse error." );
					SocketShim.usageAndExit( );
				}
			case 2:
				spam = argv[1];
			case 1:
				address = argv[0];
				break;
			default:
				SocketShim.usageAndExit( );
		}

		Date start = new Date( );
		SocketShim shim = new SocketShim( );
		shim.connect( address, port );
		shim.write( "z;" + (new Date( ).getTime( ) - start.getTime( ))/1000.0 + "\n" );
		shim.write( "a;" + (new Date( ).getTime( ) - start.getTime( ))/1000.0 + ";1\n" );

		int i = 0;
		while ( true ) {
			try {
				Thread.sleep( 100*(((i++ % 10) == 0)? 10: 1) );
			} catch ( InterruptedException e ) {
				break;
			}
			String msg = "d;" + (new Date( ).getTime( ) - start.getTime( ))/1000.0 + ";" + spam + "\n";
			shim.write( msg );
			System.err.print( msg );
			shim.flush( );
			shim.read( );
		}
	}

	public SocketShim( ) { }

	public void connect( String address, int port ) throws ShimException, UnknownHostException {
		if ( open ) {
			close( );
		}

		// Don't catch the UnknownHostException, because it may be useful at the
		// other end (ShimException is essentially a stand-in for an unrecoverable
		// error)
		InetAddress addr = InetAddress.getByName( address );

		try {
			sock = new Socket( addr, port );
		} catch ( IOException e ) {
			throw new ShimException( "Can't connect to host '" + address + ":" + port + "'. " + e.getMessage( ) );
		} catch ( IllegalArgumentException e ) {
			throw new ShimException( "Bad port number '" + port  + "'. " + e.getMessage( ) );
		}

		try {
			readStream = sock.getInputStream( );
			writeStream = sock.getOutputStream( );
		} catch ( IOException e ) {
			close( );
			throw new ShimException( "Can't produce streams. " + e.getMessage( ) );
		}

		open = true;
	}

	public String read( ) throws ShimException {
		int readSize = 0;
		try {
			readSize = readStream.available( );
		} catch ( IOException e ) {
			close( );
			throw new ShimException( "Can't check read availability. " + e.getMessage( ) );
		}
		if ( readSize > 0 ) {
			byte[] buffer = new byte[readSize];
			try {
				readSize = readStream.read( buffer, 0, readSize );
			} catch ( IOException e ) {
				close( );
				throw new ShimException( "Could not read. " + e.getMessage( ) );
			}
			return new String( buffer, 0, readSize );
		} else {
			return new String( );
		}
	}

	public void write( String message ) throws ShimException {
		try {
			writeStream.write( message.getBytes( Charset.forName("UTF-8") ) );
		} catch ( IOException e ) {
			close( );
			throw new ShimException( "Could not write. " + e.getMessage( ) );
		}
	}

	public void flush( ) throws ShimException {
		try {
			writeStream.flush( );
		} catch ( IOException e ) {
			close( );
			throw new ShimException( "Could not flush. " + e.getMessage( ) );
		}
	}

	public void close( ) {
		open = false;
		try {
			sock.close( );
		} catch ( IOException e ) { }
		// throwing here is detrimental?
	}

	public boolean isOpen( ) {
		return open;
	}
}
