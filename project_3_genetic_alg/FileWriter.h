#pragma once

#include <fstream>
#include <string>

namespace FileWriter {

	/**
	 * Writes a string to a file
	 */
	void Write(std::string filename, std::string contents) {
		std::ofstream f;
		f.open(filename);
		f << contents;
		f.close();
	}

};
